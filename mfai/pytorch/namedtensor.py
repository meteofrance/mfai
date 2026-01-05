"""
A class based NamedTensor implementation for PyTorch, inspired from the unstable PyTorch namedtensors.
"""

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from itertools import chain
from typing import Any, Sequence, Union

import einops
import torch
from tabulate import tabulate
from torch import Tensor


@dataclass(slots=True)
class TensorWrapper:
    """
    Wrapper around a torch tensor.
    We do this separated dataclass to allow lightning's introspection to see our batch size
    and move our tensors to the right device, otherwise we have this error/warning:
    "Trying to infer the `batch_size` from an ambiguous collection ..."
    """

    tensor: Tensor


class NamedTensor(TensorWrapper):
    """
    NamedTensor is a wrapper around a torch tensor
    adding several attributes :

    * a 'names' attribute with the names of the
    tensor's dimensions (like https://pytorch.org/docs/stable/named_tensor.html).

    Torch's named tensors are still experimental and subject to change.

    * a 'feature_names' attribute containing the names of the features
    along the last dimension of the tensor.

    NamedTensor can be concatenated along the last dimension
    using the | operator.
    nt3 = nt1 | nt2
    """

    SPATIAL_DIM_NAMES = ("lat", "lon", "ngrid")

    def __init__(
        self,
        tensor: Tensor,
        names: Sequence[str],
        feature_names: Sequence[str],
        feature_dim_name: str = "features",
    ):
        if len(tensor.shape) != len(names):
            raise ValueError(
                f"Number of names ({len(names)}) must match number of dimensions ({len(tensor.shape)})"
            )
        if tensor.shape[names.index(feature_dim_name)] != len(feature_names):
            raise ValueError(
                f"Number of feature names ({len(feature_names)}:{feature_names}) must match "
                f"number of features ({tensor.shape[names.index(feature_dim_name)]}) in the supplied tensor"
            )

        super().__init__(tensor)

        self.names: list[str] = list(names)
        # build lookup table for fast indexing
        self.feature_names_to_idx = {
            feature_name: idx for idx, feature_name in enumerate(feature_names)
        }
        self.feature_names: list[str] = list(feature_names)
        self.feature_dim_name = feature_dim_name

    @property
    def ndims(self) -> int:
        """
        Number of dimensions of the tensor.
        """
        return len(self.names)

    @property
    def num_spatial_dims(self) -> int:
        """
        Number of spatial dimensions of the tensor.
        """
        return len([x for x in self.names if x in self.SPATIAL_DIM_NAMES])

    @property
    def feature_dim_idx(self) -> int:
        """
        Index of the features dimension.
        """
        return self.names.index(self.feature_dim_name)

    def __str__(self) -> str:
        head = "--- NamedTensor ---\n"
        head += f"Names: {self.names}\nTensor Shape: {self.tensor.shape})\nFeatures:\n"
        table = [
            [feature, self[feature].min(), self[feature].max()]
            for feature in self.feature_names
        ]
        headers = ["Feature name", "Min", "Max"]
        table_string = str(tabulate(table, headers=headers, tablefmt="simple_outline"))
        return head + table_string

    def __or__(self, other: Union["NamedTensor", None]) -> "NamedTensor":
        """
        Concatenate two NamedTensors along the features dimension.
        """
        if other is None:
            return self

        if not isinstance(other, NamedTensor):
            raise ValueError("Can only concatenate NamedTensor with NamedTensor")

        # check features names are distinct between the two tensors
        if set(self.feature_names) & set(other.feature_names):
            raise ValueError(
                f"Feature names must be distinct between the two tensors for"
                f"unambiguous concat, self:{self.feature_names} other:{other.feature_names}"
            )

        if self.names != other.names:
            raise ValueError(
                f"NamedTensors must have the same dimension names to concatenate, self:{self.names} other:{other.names}"
            )
        try:
            return NamedTensor(
                torch.cat([self.tensor, other.tensor], dim=self.feature_dim_idx),
                self.names.copy(),
                self.feature_names + other.feature_names,
            )
        except Exception as e:
            raise ValueError(f"Error while concatenating {self} and {other}") from e

    def __ror__(self, other: Union["NamedTensor", None]) -> "NamedTensor":
        return self.__or__(other)

    @staticmethod
    def stack(
        nts: Sequence["NamedTensor"], dim_name: str, dim: int = 0
    ) -> "NamedTensor":
        """
        Stack a list of NamedTensors along a new dimension.
        """
        if len(nts) == 0:
            raise ValueError("Cannot stack an empty list of NamedTensors")
        if len(nts) == 1:
            return nts[0].clone()
        else:
            # Check features names are identical between the n named tensors
            if not all(nt.feature_names == nts[0].feature_names for nt in nts):
                raise ValueError(
                    "NamedTensors must have the same feature names to stack"
                )

            # Check that all named tensors have the same dim names
            if not all(nt.names == nts[0].names for nt in nts):
                raise ValueError(
                    "NamedTensors must have the same dimension names to stack"
                )

            # define new list of dim names, with new dim name inserted at dim
            names = nts[0].names.copy()
            names.insert(dim, dim_name)

            new_tensor = torch.stack([nt.tensor for nt in nts], dim=dim)
            return NamedTensor(new_tensor, names, nts[0].feature_names.copy())

    @staticmethod
    def concat(nts: Sequence["NamedTensor"]) -> "NamedTensor":
        """
        Safely concat a list of NamedTensors along the last dimension
        in one shot.
        """
        if len(nts) == 0:
            raise ValueError("Cannot concatenate an empty list of NamedTensors")
        if len(nts) == 1:
            return nts[0].clone()
        else:
            # Check features names are distinct between the n named tensors
            feature_names: set[str] = set()
            for nt in nts:
                if feature_names & set(nt.feature_names):
                    raise ValueError(
                        f"Feature names must be distinct between the named tensors to concat\n"
                        f"Found duplicates: {feature_names & set(nt.feature_names)}"
                    )
                feature_names |= set(nt.feature_names)

            # Check that all named tensors have the same names
            if not all(nt.names == nts[0].names for nt in nts[1:]):
                raise ValueError(
                    "NamedTensors must have the same dimension names to concatenate"
                )

            # Check that all named tensors have the same feature dimension name
            if not all(
                nt.feature_dim_name == nts[0].feature_dim_name for nt in nts[1:]
            ):
                raise ValueError(
                    "NamedTensors must have the same feature dimension name to concatenate"
                )

            # Concat in one shot
            return NamedTensor(
                torch.cat([nt.tensor for nt in nts], dim=nts[0].feature_dim_idx),
                nts[0].names.copy(),
                list(chain.from_iterable(nt.feature_names for nt in nts)),
                feature_dim_name=nts[0].feature_dim_name,
            )

    def dim_index(self, dim_name: str) -> int:
        """
        Return the index of a dimension given its name.
        """
        return self.names.index(dim_name)

    def clone(self) -> "NamedTensor":
        """Clone (with a deepcopy) the NamedTensor."""
        return NamedTensor(
            tensor=deepcopy(self.tensor).to(self.tensor.device),
            names=self.names.copy(),
            feature_names=self.feature_names.copy(),
        )

    def __getitem__(self, feature_name: str) -> Tensor:
        """
        Get one feature from the features dimension of the tensor by name.
        The returned tensor has the same number of dimensions as the original tensor.
        """
        try:
            return self.tensor.select(
                self.names.index(self.feature_dim_name),
                self.feature_names_to_idx[feature_name],
            ).unsqueeze(self.names.index(self.feature_dim_name))
        except KeyError:
            raise ValueError(
                f"Feature {feature_name} not found in {self.feature_names}"
            )

    def type_(self, new_type: str | torch.dtype) -> None:
        """
        Modify the type of the underlying torch tensor
        by calling torch's .type method

        in_place operation for this class, the internal
        tensor is replaced by the new one.
        """
        self.tensor = self.tensor.type(new_type)

    def flatten_(
        self, flatten_dim_name: str, start_dim: int = 0, end_dim: int = -1
    ) -> None:
        """
        Flatten the underlying tensor from start_dim to end_dim.
        Deletes flattened dimension names and insert
        the new one.
        """
        self.tensor = torch.flatten(self.tensor, start_dim, end_dim)

        # Remove the flattened dimensions from the names
        # and insert the replacing one
        end_dim = len(self.names) if end_dim == -1 else end_dim
        self.names = (
            self.names[:start_dim] + [flatten_dim_name] + self.names[end_dim + 1 :]
        )

    def unflatten_(
        self, dim: int, unflattened_size: torch.Size, unflatten_dim_name: Sequence[str]
    ) -> None:
        """
        Unflatten the dimension dim of the underlying tensor.
        Insert unflattened_size dimension instead
        """
        self.tensor = self.tensor.unflatten(dim, unflattened_size)
        self.names = self.names[:dim] + [*unflatten_dim_name] + self.names[dim + 1 :]

    def squeeze_(self, dim_name: Union[Sequence[str], str]) -> None:
        """
        Squeeze the underlying tensor along the dimension(s)
        given its/their name(s).
        """
        if isinstance(dim_name, str):
            dim_name = [dim_name]
        dim_indices = [self.names.index(name) for name in dim_name]
        self.tensor = torch.squeeze(self.tensor, dim=dim_indices)
        for name in dim_name:
            self.names.remove(name)

    def unsqueeze_(self, dim_name: str, dim_index: int) -> None:
        """
        Insert a new dimension dim_name of size 1 at dim_index
        """
        self.tensor = torch.unsqueeze(self.tensor, dim_index)
        self.names.insert(dim_index, dim_name)

    def select_dim(self, dim_name: str, index: int) -> "NamedTensor":
        """
        Return the tensor indexed along the dimension dim_name
        with the index index.
        The given dimension is removed from the tensor.
        See https://pytorch.org/docs/stable/generated/torch.select.html
        """
        if dim_name == self.feature_dim_name:
            raise ValueError(
                "Impossible to select the feature dimension of a NamedTensor."
            )

        return NamedTensor(
            self.tensor.select(self.names.index(dim_name), index),
            self.names[: self.names.index(dim_name)]
            + self.names[self.names.index(dim_name) + 1 :],
            self.feature_names,
            feature_dim_name=self.feature_dim_name,
        )

    def select_tensor_dim(self, dim_name: str, index: int) -> Tensor:
        """
        Same as select_dim but returns a Tensor.
        Allows the selection of the feature dimension.
        """
        return self.tensor.select(self.names.index(dim_name), index)

    def index_select_dim(self, dim_name: str, indices: Sequence[int]) -> "NamedTensor":
        """
        Return the tensor indexed along the dimension dim_name
        with the indices tensor.
        The returned tensor has the same number of dimensions as the original tensor (input).
        The dimth dimension has the same size as the length of index; other dimensions have
        the same size as in the original tensor.
        See https://pytorch.org/docs/stable/generated/torch.index_select.html
        """
        return NamedTensor(
            self.tensor.index_select(
                self.names.index(dim_name),
                Tensor(indices).type(torch.int64).to(self.device),
            ),
            self.names,
            (
                self.feature_names
                if dim_name != self.feature_dim_name
                else [self.feature_names[i] for i in indices]
            ),
            feature_dim_name=self.feature_dim_name,
        )

    def index_select_tensor_dim(
        self, dim_name: str, indices: Sequence[int]
    ) -> torch.Tensor:
        """
        Same as index_select_dim but returns a torch.tensor, but returns a Tensor.
        """
        return self.tensor.index_select(
            self.names.index(dim_name),
            Tensor(indices).type(torch.int64).to(self.device),
        )

    def dim_size(self, dim_name: str) -> int:
        """
        Return the size of a dimension given its name.
        """
        try:
            return self.tensor.size(self.names.index(dim_name))
        except ValueError as ve:
            raise ValueError(f"Dimension {dim_name} not found in {self.names}") from ve

    @property
    def spatial_dim_idx(self) -> list[int]:
        """
        Return the indices of the spatial dimensions in the tensor.
        """
        return sorted(
            self.names.index(name)
            for name in set(self.SPATIAL_DIM_NAMES).intersection(set(self.names))
        )

    def unsqueeze_and_expand_from_(self, other: "NamedTensor") -> None:
        """
        Unsqueeze and expand the tensor to have the same number of spatial dimensions
        as another NamedTensor.
        Injects new dimensions where the missing names are.
        """
        missing_names = set(other.names) - set(self.names)
        missing_names &= set(self.SPATIAL_DIM_NAMES)

        if missing_names:
            index_to_unsqueeze = [
                (name, other.names.index(name)) for name in missing_names
            ]
            for name, idx in sorted(index_to_unsqueeze, key=lambda x: x[1]):
                self.tensor = torch.unsqueeze(self.tensor, idx)
                self.names.insert(idx, name)

            expander = []
            for _, name in enumerate(self.names):
                expander.append(other.dim_size(name) if name in missing_names else -1)

            self.tensor = self.tensor.expand(*expander)

    def iter_dim(self, dim_name: str) -> Iterable["NamedTensor"]:
        """
        Iterate over the tensor along a given dimension.
        """
        for i in range(self.dim_size(dim_name)):
            yield self.select_dim(dim_name, i)

    def iter_tensor_dim(self, dim_name: str) -> Iterable[Tensor]:
        """
        Iterate over the tensor along a given dimension.
        """
        for i in range(self.dim_size(dim_name)):
            yield self.select_tensor_dim(dim_name, i)

    def rearrange_(self, einops_str: str) -> None:
        """
        Rearrange in place the underlying tensor dimensions using einops syntax.
        For now only supports re-ordering of dimensions.
        """
        old_dims_str, new_dims_str = einops_str.split("->")
        old_dims = old_dims_str.split(" ")[:-1]
        new_dims = new_dims_str.split(" ")[1:]
        # check that the number of dims and dim names match
        if not set(self.names) == set(old_dims) == set(new_dims):
            raise ValueError(
                f"Dimensions in rearrange_ {old_dims} do not match tensor dimensions {self.names}"
            )
        self.tensor = einops.rearrange(self.tensor, einops_str)
        self.names = new_dims

    @staticmethod
    def new_like(tensor: Tensor, other: "NamedTensor") -> "NamedTensor":
        """
        Create a new NamedTensor with the same names and feature names as another NamedTensor.
        """
        return NamedTensor(tensor, other.names.copy(), other.feature_names.copy())

    @staticmethod
    def expand_to_batch_like(tensor: Tensor, other: "NamedTensor") -> "NamedTensor":
        """
        Create a new NamedTensor with the same names and feature names as another NamedTensor
        with an extra first dimension called 'batch' using the supplied tensor.
        Supplied new 'batched' tensor must have one more dimension than other.
        """
        names = ["batch"] + other.names
        if tensor.dim() != len(names):
            raise ValueError(
                f"Tensor dim {tensor.dim()} must match number of names {len(names)} with extra batch dim"
            )
        return NamedTensor(tensor, ["batch"] + other.names, other.feature_names.copy())

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def pin_memory_(self) -> None:
        """
        'In place' operation to pin the underlying tensor to memory.
        """
        self.tensor = self.tensor.pin_memory()

    def to_(self, *args: Any, **kwargs: Any) -> None:
        """
        'In place' operation to call torch's 'to' method on the underlying tensor.
        """
        self.tensor = self.tensor.to(*args, **kwargs)

    @staticmethod
    def collate_fn(
        batch: Sequence["NamedTensor"],
        pad_dims: tuple[str, ...] | tuple[()] = (),
        pad_value: int | float = 0,
    ) -> "NamedTensor":
        """
        Collate a list of NamedTensors into a batched single NamedTensor.
        Optionnally pads the dimensions specified in pad_dims with pad_value.
        """
        if len(batch) == 0:
            raise ValueError("Cannot collate an empty list of NamedTensors")
        if len(batch) == 1:
            # add batch dim to the single namedtensor (in place operation)
            batch[0].unsqueeze_(dim_name="batch", dim_index=0)
            return batch[0]
        else:
            if len(pad_dims):
                # Find the maximum size for each dimension in pad_dims looking
                # at each tensor of each NamedTensor in the batch
                max_sizes: dict[str, int] = {
                    dim: max(nt.dim_size(dim) for nt in batch) for dim in pad_dims
                }

                # Pad each tensor in the batch to the maximum size, for each dim we pad after the data
                padded_batch = []
                for nt in batch:
                    padded_shape = list(nt.tensor.shape)
                    do_pad: bool = False

                    for dim in pad_dims:
                        dim_idx = nt.dim_index(dim)
                        if nt.dim_size(dim) < max_sizes[dim]:
                            padding_size = max_sizes[dim] - nt.dim_size(dim)
                            padded_shape[dim_idx] += padding_size
                            do_pad = True

                    if do_pad:
                        padded_tensor = nt.tensor.new_full(
                            padded_shape,
                            pad_value,
                        )

                        slicer = tuple(slice(0, nt.dim_size(dim)) for dim in nt.names)

                        padded_tensor[slicer] = nt.tensor
                    else:
                        padded_tensor = nt.tensor.clone()

                    padded_batch.append(NamedTensor.new_like(padded_tensor, nt))

                return NamedTensor.stack(padded_batch, dim_name="batch", dim=0)

            else:
                return NamedTensor.stack(batch, dim_name="batch", dim=0)
