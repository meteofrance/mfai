# mfai.NamedTensor

```python
class mfai.NamedTensor(tensor: Tensor, names: List[str], feature_names: List[str], feature_dim_name: str = "features")
```

The [**NamedTensor**](mfai/pytorch/namedtensor.py#L28) class is a wrapper around a PyTorch tensor with additionnal attributes and methods, it allows us to pass consistent object linking data and metadata with extra utility methods (concat along features dimension, flatten in place, ...).

## Parameters
- **tensor (torch.Tensor):** The tensor to wrap.
- **names (list of str):**  Names of the tensor's dimensions.
- **feature_names (list of str):** Names of the features along the 'feature_dim_name' of the tensor.
- **feature_dim_name (str, optional):** Name of the feature dimension.

## Attributes

- **`device` (torch.device):** Device where the Tensor is stored.
- **`feature_dim_name` (str):** Name of the feature dimension.
- **`feature_names` (list of str):** Names of the features along the 'feature_dim_name' of the tensor.
- **`feature_names_to_idx` (dict):** Dictionary mapping feature names to their indices.
- **`names` (list of str):** Names of the tensor's dimensions.
- **`ndims` (int):** Number of dimensions of the tensor.
- **`num_spatial_dims` (int):** Number of spatial dimensions of the tensor.

- **`spatial_dim_idx` (list of int):** Indices of the spatial dimensions in the tensor.
- **`tensor` (torch.Tensor):** The wrapped tensor.

## Methods

| Method  | Description  |
| :---:   | :---: |
| `clone`() | Clone with a deepcopy. |
| `collate_fn`(batch, pad_dims, pad_value) | Collates a batch of `NamedTensor` instances into a single `NamedTensor`. Optionnaly padding on the desired dimensions with `pad_value`|
| `concat`(nts) | Concatenates a list of `NamedTensor` instances along the features dimension. |
| `dim_index`(dim_name) | Return the index of a dimension given its name. |
| `dim_size`(dim_name) | Returns the size of a dimension given its name. |
| `expand_to_batch_like`(tensor, other) | Creates a new `NamedTensor` with the same names and feature names as another `NamedTensor` with an extra first dimension called 'batch' using the supplied tensor. |
| `flatten_`(new_dim_name, start_dim, end_dim) | Flatten in place the underlying tensor from start_dim to end_dim. Deletes flattened dimension names and insert the new one. |
| `index_select_dim`(dim_name, indices) | Returns the tensor indexed along the dimension `dim_name` with the indices tensor. The returned tensor has the same number of dimensions as the original tensor. The `dim_name` dimension has the same size as the length of `indices`; other dimensions have the same size as in the original tensor. |
| `index_select_tensor_dim`(dim_name, indices) | Same as `index_select_dim()` but returns a `torch.Tensor`. |
| `new_like`(tensor, other) | Creates a new `NamedTensor` with the same names but different data. |
| `iter_dim`(dim_name) | Iterates over the specified dimension, yielding `NamedTensor` instances. |
| `iter_tensor_dim`(dim_name) | Iterates over the specified dimension, yielding Tensor instances. |
| `pin_memory_`() | In place operation to pin the underlying tensor to memory. |

| `rearrange_`(einops_str) | Rearranges the tensor in place using einops-like syntax. |
| `select_dim`(dim_name, index) | Returns the `NamedTensor` indexed along the dimension `dim_name` with the desired index. The given dimension is removed from the tensor.This method can not select the feature dimension. |
| `select_tensor_dim`(dim_name, index) | Return the Tensor indexed along the dimension dim_name with the index index. Allows the selection of the feature dimension. Allows the selection of the feature dimension. |
| `squeeze_`(dim_name) | Squeeze the underlying tensor along the dimension(s) given its/their name(s). |
| `stack`(tensors, dim_name, dim) | Stacks a list of `NamedTensor` instances along a new dimension. |
| `to_`(*args, **kwargs) | In place operation to call torch's 'to' method on the underlying tensor. |
| `type_`(new_type) | Modifies the type of the underlying torch tensor by calling torch's `.type` method. This is an in-place operation for this class, the internal tensor is replaced by the new one. |
| `unflatten_`(dim, unflattened_size, unflatten_dim_name) | Unflatten the dimension dim of the underlying tensor. Insert unflattened_size dimension instead. |
| `unsqueeze_`(dim_name, dim_index) | Insert a new dimension dim_name of size 1 at dim_index. |
| `unsqueeze_and_expand_from_`(other) | Unsqueeze and expand the tensor to have the same number of spatial dimensions as another `NamedTensor`. Injects new dimensions where the missing names are. |


## Special Methods

| Method  | Description  |
| :---:   | :---: |
| `__getitem__`(feature_name) | Get one feature from the features dimension of the tensor by name. The returned tensor has the same number of dimensions as the original tensor. |
| `__or__`(other) | Concatenate two `NamedTensor` along the features dimension. |
| `__str__`(other) | Returns a string representation of the `NamedTensor` with useful statistics. |


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

# Collate a batch with zero padding on lat, lon dimensions

nt1 = NamedTensor(
    torch.rand(128, 128, 3),
    names=["lat", "lon", "features"],
    feature_names=["u", "v", "t2m"],
)

nt2 = NamedTensor(
    torch.rand(256, 256, 3),
    names=["lat", "lon", "features"],
    feature_names=["u", "v", "t2m"],
)

collated_nt_padded = NamedTensor.collate_fn([nt1, nt2], pad_dims=("lat", "lon"), pad_value=0.0)

```

For more details, refer to the **NamedTensor** class in `mfai/pytorch/namedtensor.py`.