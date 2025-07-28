# NamedTensor

The [**NamedTensor**](mfai/pytorch/namedtensor.py#L28) class is a wrapper around a PyTorch tensor with additionnal attributes and methods, it allows us to pass consistent object linking data and metadata with extra utility methods (concat along features dimension, flatten in place, ...).

Table of Contents:
- [Attributes](#attributes)
- [Methods](#methods)
  - [\_\_init\_\_](#init)
  - [expand_to_batch_like](#expand_to_batch_like)
  - [spatial_dim_idx](#spatial_dim_idx)
  - [dim_size](#dim_size)
  - [\_\_str\_\_](#__str__)
  - [select](#select)
  - [index_select](#index_select)
  - [new_like](#new_like)
  - [flatten_](#flatten_)
  - [rearrange_](#rearrange_)
  - [collate_fn](#collate_fn)
  - [concat](#concat)
  - [stack](#stack)
  - [iter_dim](#iter_dim)
  - [type_](#type_)
- [Example Usage](#example-usage)
    - [Instantiation](#instantiation)
    - [Concatenation, Indexing, New Like, Flatten, Rearrange](#concatenation-indexing-new-like-flatten-rearrange)
    - [Selection and Index Selection](#selection-and-index-selection)
    - [Iteration](#iteration)
    - [Collation](#collation)


## Attributes

- **`device`**: The name of the device where the Tensor is stored.
- **`feature_dim_name`**: The name of the feature dimension.
- **`feature_names`**: A list of names for the features along the last dimension of the tensor.
- **`feature_names_to_idx`**: A dictionary mapping feature names to their indices.
- **`names`**: A list of names for the tensor's dimensions.
- **`ndims`**: Number of dimensions of the tensor.
- **`num_spatial_dims`**: Number of spatial dimensions of the tensor.
- **`tensor`**: The `Tensor`.

## Methods

### `__init__(self, tensor: Tensor, names: List[str], feature_names: List[str], feature_dim_name: str = "features")`
Initializes a **NamedTensor** instance.

### `__getitem__(self, feature_name: str) -> Tensor`
Get one feature from the features dimension of the tensor by name.
The returned tensor has the same number of dimensions as the original tensor.

### `__or__(self, other: Union["NamedTensor", None]) -> "NamedTensor"`
Concatenate two NamedTensors along the features dimension.

### `__str__(self)`
Returns a string representation of the **NamedTensor** with useful statistics.

### `clone(self) -> "NamedTensor"`
Clone (with a deepcopy) the NamedTensor.

### `collate_fn(cls, batch: List["NamedTensor"]) -> "NamedTensor"`
Collates a batch of **NamedTensor** instances into a single **NamedTensor**.

### `concat(cls, tensors: List["NamedTensor"]) -> "NamedTensor"`
Concatenates a list of **NamedTensor** instances along the features dimension.

### `dim_index(self, dim_name: str) -> int`
Return the index of a dimension given its name.

### `dim_size(self, dim_name: str) -> int`
Returns the size of a dimension given its name.

### `expand_to_batch_like(tensor: Tensor, other: "NamedTensor") -> "NamedTensor"`
Creates a new **NamedTensor** with the same names and feature names as another **NamedTensor** with an extra first dimension called 'batch' using the supplied tensor.

### `flatten_(self, new_dim_name: str, dim1: int, dim2: int)`
Flattens in place the dimensions `dim1` and `dim2` and renames the new dimension to `new_dim_name`.

### `index_select_dim(self, dim_name: str, indices: Tensor) -> "NamedTensor"`
Returns the tensor indexed along the dimension `dim_name` with the indices tensor. The returned tensor has the same number of dimensions as the original tensor. The `dim_name` dimension has the same size as the length of `indices`; other dimensions have the same size as in the original tensor.
See https://pytorch.org/docs/stable/generated/torch.index_select.html.

### `index_select_tensor_dim(self, dim_name: str, indices: Tensor) -> Tensor`
Same as [index_select_dim](#index_select_dimself-dim_name-str-indices-torchtensor---namedtensor) but returns a Tensor.

### `new_like(cls, tensor: Tensor, other: "NamedTensor") -> "NamedTensor"`
Creates a new **NamedTensor** with the same names but different data.

### `iter_dim(self, dim_name: str) -> Iterable["NamedTensor"]`
Iterates over the specified dimension, yielding **NamedTensor** instances.

### `iter_tensor_dim(self, dim_name: str) -> Iterable[Tensor]`
Iterates over the specified dimension, yielding **Tensor** instances.

### `pin_memory_(self) -> None`
'In place' operation to pin the underlying tensor to memory.

### `rearrange_(self, pattern: str)`
Rearranges the tensor in place using einops-like syntax.

### `select_dim(self, dim_name: str, index: int) -> "NamedTensor"`
Returns the NamedTensor indexed along the dimension `dim_name` with the desired index. The given dimension is removed from the tensor.
This method can not select the feature dimension.

### `select_tensor_dim(self, dim_name: str, index: int) -> Tensor`
Return the Tensor indexed along the dimension dim_name with the index index. Allows the selection of the feature dimension.
Allows the selection of the feature dimension.

### `spatial_dim_idx(self) -> List[int]`
Returns the indices of the spatial dimensions in the tensor.

### `squeeze_(self, dim_name: Union[List[str], str]) -> None`
Squeeze the underlying tensor along the dimension(s) given its/their name(s).

### `stack(cls, tensors: List["NamedTensor"], dim_name: str, dim: int) -> "NamedTensor"`
Stacks a list of **NamedTensor** instances along a new dimension.

### `to_(self, *args: Any, **kwargs: Any) -> None`
'In place' operation to call torch's 'to' method on the underlying tensor.

### `type_(self, new_type)`
Modifies the type of the underlying torch tensor by calling torch's `.type` method. This is an in-place operation for this class, the internal tensor is replaced by the new one.

### `unflatten_(self, dim: int, unflattened_size: torch.Size, unflatten_dim_name: Sequence[str]) -> None`
Unflatten the dimension dim of the underlying tensor. Insert unflattened_size dimension instead.

### `unsqueeze_(self, )`
Insert a new dimension dim_name of size 1 at dim_index.

### `unsqueeze_and_expand_from_(self, other: "NamedTensor") -> None`
Unsqueeze and expand the tensor to have the same number of spatial dimensions as another NamedTensor.
Injects new dimensions where the missing names are.

## Example Usage

### Instantiation

In the following example, we create a **NamedTensor** from a PyTorch tensor with the following dimensions: batch, lat, lon, features. We also provide the names of the dimensions and the names of the features using respectively the `names` and `feature_names` arguments.

```python
import torch
from torch import Tensor
from mfai.pytorch.namedtensor import NamedTensor

tensor = torch.rand(4, 256, 256, 3)

nt = NamedTensor(
    tensor,
    names=["batch", "lat", "lon", "features"],
    feature_names=["u", "v", "t2m"],
)
```

### Concatenation, Indexing, New Like, Flatten, Rearrange

```python
import torch
from torch import Tensor
from mfai.pytorch.namedtensor import NamedTensor

nt1 = NamedTensor(
    torch.rand(4, 256, 256, 3),
    names=["batch", "lat", "lon", "features"],
    feature_names=["u", "v", "t2m"],
)

nt2 = NamedTensor(
    torch.rand(4, 256, 256, 1),
    names=["batch", "lat", "lon", "features"],
    feature_names=["q"],
)

# Concatenate along the features dimension
nt3 = nt1 | nt2

# Index by feature name
u_feature = nt3["u"]

# Create a new NamedTensor with the same names but different data
nt4 = NamedTensor.new_like(torch.rand(4, 256, 256, 4), nt3)

# Flatten in place the lat and lon dimensions and rename the new dim to 'ngrid'
nt3.flatten_("ngrid", 1, 2)

# String representation of the NamedTensor yields useful statistics
print(nt3)

# Rearrange in place using einops-like syntax
nt3.rearrange_("batch ngrid features -> batch features ngrid")
```

### Selection and Index Selection

```python
nt = NamedTensor(
    torch.rand(4, 256, 256, 3),
    names=["batch", "lat", "lon", "features"],
    feature_names=["u", "v", "t2m"],
)

# Return the tensor indexed along the dimension dim_name with the desired index. The given dimension is removed from the tensor.
selected_named_tensor = nt.select_dim("lat", 0)
selected_tensor = nt.select_tensor_dim("lat", 0)
assert selected_named_tensor.tensor == selected_tensor

# Return the tensor indexed along the dimension dim_name with the indices tensor. The returned tensor has the same number of dimensions as the original tensor (input). The dim_name dimension has the same size as the length of indices; other dimensions have the same size as in the original tensor.
indexed_tensor = nt.index_select_dim("features", torch.tensor([0, 2]))
```

### Iteration

```python
nt = NamedTensor(
    torch.rand(4, 256, 256, 3),
    names=["batch", "lat", "lon", "features"],
    feature_names=["u", "v", "t2m"],
)


# Iterate over the 'batch' dimension, yielding NamedTensor instances
for named_tensor in nt.iter_dim("batch"):
    print(named_tensor)

# Iterate over the 'batch' dimension, yielding Tensor instances
for tensor in nt.iter_tensor_dim("batch"):
    print(tensor.shape)
```

### Collation

```python
nt1 = NamedTensor(
    torch.rand(256, 256, 3),
    names=["lat", "lon", "features"],
    feature_names=["u", "v", "t2m"],
)

nt2 = NamedTensor(
    torch.rand(256, 256, 3),
    names=["lat", "lon", "features"],
    feature_names=["u", "v", "t2m"],
)

# Collate a batch of NamedTensor instances into a single NamedTensor
collated_nt = NamedTensor.collate_fn([nt1, nt2])
print(collated_nt)
```

For more details, refer to the **NamedTensor** class in `mfai/pytorch/namedtensor.py`.