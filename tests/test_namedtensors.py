from copy import copy

import pytest
import torch

from mfai.pytorch.namedtensor import NamedTensor


def test_named_tensor() -> None:
    """
    Test NamedTensor class.
    """
    # Create a tensor
    tensor = torch.rand(3, 256, 256, 50)

    # Create a NamedTensor
    nt = NamedTensor(
        tensor,
        names=["batch", "lat", "lon", "features"],
        feature_names=[f"feature_{i}" for i in range(50)],
    )
    # Test
    assert nt.names == ["batch", "lat", "lon", "features"]
    assert nt.tensor.shape == (3, 256, 256, 50)

    # Test dim sizes
    assert nt.dim_size("batch") == 3
    assert nt.dim_size("lat") == 256
    assert nt.dim_size("lon") == 256
    assert nt.dim_size("features") == 50

    # Create a NamedTensor with 'names' and 'feature_names' as tuples
    nt_tuple = NamedTensor(
        tensor,
        names=tuple(["batch", "lat", "lon", "features"]),
        feature_names=tuple([f"feature_{i}" for i in range(50)]),
    )
    assert nt == nt_tuple

    nt2 = nt.clone()

    # Concat should raise because of feature names collision
    with pytest.raises(ValueError):
        nt | nt2

    nt3 = NamedTensor(
        torch.rand(3, 256, 256, 50),
        names=["batch", "lat", "lon", "features"],
        feature_names=[f"levels_{i}" for i in range(50)],
    )

    # Test | operator (concatenation)
    # the number of features must match the sum of the feature names of the two tensors
    nt_cat = nt | nt3
    assert nt_cat.tensor.shape == (3, 256, 256, 100)

    # Test | operator (concatenation)
    # last dim name does not mach => ValueError
    nt4 = NamedTensor(
        torch.rand(3, 256, 256, 50),
        names=["batch", "lat", "lon", "levels"],
        feature_names=[f"feature_{i}" for i in range(50)],
        feature_dim_name="levels",
    )
    assert nt4.spatial_dim_idx == [1, 2]
    with pytest.raises(ValueError):
        nt | nt4

    # different number of dims => ValueError
    nt5 = NamedTensor(
        torch.rand(3, 256, 50),
        names=["batch", "lat", "levels"],
        feature_names=[f"feature_{i}" for i in range(50)],
        feature_dim_name="levels",
    )
    with pytest.raises(ValueError):
        nt | nt5

    # missing feature with __getitem__ lookup => ValueError
    with pytest.raises(ValueError):
        nt["feature_50"]

    # valid feature name should return tensor of the right shape (unsqueezed on feature dim)
    f = nt["feature_0"]
    assert f.shape == (3, 256, 256, 1)

    # test expanding a lower dim NamedTensor to a higher dim NamedTensor
    nt6 = NamedTensor(
        torch.rand(3, 10),
        names=["batch", "features"],
        feature_names=[f"f_{i}" for i in range(10)],
    )
    nt6.unsqueeze_and_expand_from_(nt)
    assert nt6.tensor.shape == (3, 256, 256, 10)
    assert nt6.names == ["batch", "lat", "lon", "features"]

    # Test flattenting all dims (default behavior)
    nt6_copy = copy(nt6)
    nt6.flatten_("flat_dim")
    nt6_copy.flatten_("flat_dim", 0, 3)
    assert nt6 == nt6_copy
    assert nt6.names == ["flat_dim"]

    # test flattening lat,lon to ndims to simulate gridded data with 2D spatial dims into a GNN
    nt7 = NamedTensor(
        torch.rand(3, 256, 256, 10),
        names=["batch", "lat", "lon", "features"],
        feature_names=[f"feature_{i}" for i in range(10)],
    )
    nt7.flatten_("ngrid", 1, 2)
    assert nt7.tensor.shape == (3, 256**2, 10)
    assert nt7.names == ["batch", "ngrid", "features"]
    assert nt7.spatial_dim_idx == [1]

    # test creating a NamedTensor from another NamedTensor
    new_nt = NamedTensor.new_like(torch.rand(3, 256, 256, 50), nt)

    # make sure our __str__ works
    print(new_nt)

    # it must raise ValueError if wrong number of dims
    with pytest.raises(ValueError):
        NamedTensor.new_like(torch.rand(3, 256, 256), nt)

    # it must raise ValueError if wrong number of feature names versus last dim size
    with pytest.raises(ValueError):
        NamedTensor(
            torch.rand(3, 256, 256, 2),
            names=["batch", "lat", "lon", "features"],
            feature_names=[f"f_{i}" for i in range(5)],
        )

    # test one shot concat of multiple NamedTensors
    nt7 = NamedTensor(
        torch.rand(3, 256, 256, 10),
        names=["batch", "lat", "lon", "features"],
        feature_names=[f"feature_{i}" for i in range(10)],
    )
    nt8 = NamedTensor(
        torch.rand(3, 256, 256, 10),
        names=["batch", "lat", "lon", "features"],
        feature_names=[f"v_{i}" for i in range(10)],
    )
    nt9 = NamedTensor(
        torch.rand(3, 256, 256, 10),
        names=["batch", "lat", "lon", "features"],
        feature_names=[f"u_{i}" for i in range(10)],
    )

    nt_cat = NamedTensor.concat([nt7, nt8, nt9])

    assert nt_cat.tensor.shape == (3, 256, 256, 30)
    assert nt_cat.feature_names == [f"feature_{i}" for i in range(10)] + [
        f"v_{i}" for i in range(10)
    ] + [f"u_{i}" for i in range(10)]
    assert nt_cat.names == ["batch", "lat", "lon", "features"]

    # test stack of multiple NamedTensors
    nt10 = NamedTensor(
        torch.rand(3, 256, 256, 10),
        names=["timesteps", "lat", "lon", "features"],
        feature_names=[f"feature_{i}" for i in range(10)],
    )
    nt11 = NamedTensor(
        torch.rand(3, 256, 256, 10),
        names=["timesteps", "lat", "lon", "features"],
        feature_names=[f"feature_{i}" for i in range(10)],
    )
    nt12 = NamedTensor(
        torch.rand(3, 256, 256, 10),
        names=["timesteps", "lat", "lon", "features"],
        feature_names=[f"feature_{i}" for i in range(10)],
    )
    nt_stack = NamedTensor.stack([nt10, nt11, nt12], "level", dim=3)
    assert nt_stack.tensor.shape == (3, 256, 256, 3, 10)
    assert nt_stack.feature_names == [f"feature_{i}" for i in range(10)]
    assert nt_stack.names == ["timesteps", "lat", "lon", "level", "features"]

    # test collate of multiple NamedTensors
    nt_collate = NamedTensor.collate_fn([nt10, nt11, nt12])
    assert nt_collate.tensor.shape == (3, 3, 256, 256, 10)
    assert nt_collate.feature_names == [f"feature_{i}" for i in range(10)]
    assert nt_collate.names == ["batch", "timesteps", "lat", "lon", "features"]

    # test collate of multiple NamedTensors with pad_dims
    nt_small = NamedTensor(
        torch.rand(3, 64, 64, 10),
        names=["timesteps", "lat", "lon", "features"],
        feature_names=[f"feature_{i}" for i in range(10)],
    )
    nt_medium = NamedTensor(
        torch.rand(3, 128, 128, 10),
        names=["timesteps", "lat", "lon", "features"],
        feature_names=[f"feature_{i}" for i in range(10)],
    )
    nt_big = NamedTensor(
        torch.rand(3, 256, 256, 10),
        names=["timesteps", "lat", "lon", "features"],
        feature_names=[f"feature_{i}" for i in range(10)],
    )
    nt_padded_collated = NamedTensor.collate_fn(
        [nt_small, nt_medium, nt_big], pad_dims=("lat", "lon"), pad_value=666
    )
    assert nt_padded_collated.tensor.shape == (3, 3, 256, 256, 10)

    # check original values are preserved top left of the tensor for each item in the batch
    assert torch.all(nt_padded_collated.tensor[0, :, :64, :64, :] == nt_small.tensor)
    assert torch.all(nt_padded_collated.tensor[1, :, :128, :128, :] == nt_medium.tensor)
    assert torch.all(nt_padded_collated.tensor[2, :, :, :, :] == nt_big.tensor)

    # check padded value is as expected
    assert torch.all(nt_padded_collated.tensor[0, :, 64:, 64:, :] == 666)
    assert torch.all(nt_padded_collated.tensor[1, :, 128:, 128:, :] == 666)

    # checks that requesting padding on all the same size tensor works (does not pad)
    nt_a = NamedTensor(
        torch.rand(3, 128, 128, 10),
        names=["timesteps", "lat", "lon", "features"],
        feature_names=[f"feature_{i}" for i in range(10)],
    )
    nt_b = NamedTensor(
        torch.rand(3, 128, 128, 10),
        names=["timesteps", "lat", "lon", "features"],
        feature_names=[f"feature_{i}" for i in range(10)],
    )
    nt_padded_collated = NamedTensor.collate_fn(
        [nt_a, nt_b], pad_dims=("lat", "lon"), pad_value=666
    )
    assert nt_padded_collated.tensor.shape == (2, 3, 128, 128, 10)
    assert torch.all(nt_padded_collated.tensor[0, :, :, :, :] == nt_a.tensor)
    assert torch.all(nt_padded_collated.tensor[1, :, :, :, :] == nt_b.tensor)

    # Checks requesting padding on non existing dim breaks
    with pytest.raises(ValueError):
        nt_padded_collated = NamedTensor.collate_fn(
            [nt_a, nt_b], pad_dims=("lat", "fake"), pad_value=666
        )

    # test a features dim in the middle of the tensor (not last dim)

    nt13 = NamedTensor(
        torch.rand(3, 50, 256, 256),
        names=["batch", "features", "lat", "lon"],
        feature_names=[f"feature_{i}" for i in range(50)],
        feature_dim_name="features",
    )

    # Now concat with another tensor with the same feature dim name and dim index
    nt14 = NamedTensor(
        torch.rand(3, 50, 256, 256),
        names=["batch", "features", "lat", "lon"],
        feature_names=[f"feature_k{i}" for i in range(50)],
        feature_dim_name="features",
    )

    nt_cat = nt13 | nt14
    assert nt_cat.tensor.shape == (3, 100, 256, 256)
    assert nt_cat.feature_names == [f"feature_{i}" for i in range(50)] + [
        f"feature_k{i}" for i in range(50)
    ]
    assert nt_cat.names == ["batch", "features", "lat", "lon"]
    assert nt_cat.spatial_dim_idx == [2, 3]
    assert nt_cat.feature_dim_name == "features"

    feature_tensor = nt_cat["feature_0"]
    assert feature_tensor.shape == (3, 1, 256, 256)

    # Test iteration on batch dimension
    for i, nt_dim in enumerate(nt.iter_dim("batch")):
        assert nt_dim.tensor.shape == (256, 256, 50)
        assert nt_dim.names == ["lat", "lon", "features"]

    # Test iteration on batch dimension returning bare tensors
    for i, nt_tensor_dim in enumerate(nt.iter_tensor_dim("batch")):
        assert nt_tensor_dim.shape == (256, 256, 50)

    assert i == 2

    nt7 = NamedTensor(
        torch.rand(3, 256, 256, 10),
        names=["batch", "lat", "lon", "features"],
        feature_names=[f"feature_{i}" for i in range(10)],
    )
    nt8 = NamedTensor(
        torch.rand(3, 256, 256, 10),
        names=["batch", "lat", "lon", "features"],
        feature_names=[f"v_{i}" for i in range(10)],
    )
    nt9 = NamedTensor(
        torch.rand(3, 256, 256, 10),
        names=["batch", "lat", "lon", "features"],
        feature_names=[f"u_{i}" for i in range(10)],
    )
    nt_cat = NamedTensor.concat([nt7, nt8, nt9])
    assert nt_cat.tensor.shape == (3, 256, 256, 30)
    assert nt_cat.feature_names == [f"feature_{i}" for i in range(10)] + [
        f"v_{i}" for i in range(10)
    ] + [f"u_{i}" for i in range(10)]
    assert nt_cat.names == ["batch", "lat", "lon", "features"]

    # test unsqueeze_
    nt_cat.unsqueeze_("new_dim", 1)
    assert nt_cat.tensor.shape == (3, 1, 256, 256, 30)
    assert nt_cat.names == ["batch", "new_dim", "lat", "lon", "features"]

    # test squeeze_
    nt_cat.squeeze_("new_dim")
    assert nt_cat.tensor.shape == (3, 256, 256, 30)
    assert nt_cat.names == ["batch", "lat", "lon", "features"]

    # test selecting bare tensor features dimension
    t = nt_cat.select_tensor_dim(nt_cat.feature_dim_name, 0)
    assert t.shape == torch.Size([3, 256, 256])

    # test select_dim along the lat dim
    nt = nt_cat.select_dim("lat", 128)
    assert nt.tensor.shape == (3, 256, 30)

    # test select_tensor_dim along the lat dim
    t = nt_cat.select_tensor_dim("lat", 128)
    assert t.shape == (3, 256, 30)

    # test index_select_dim
    nt = nt_cat.index_select_dim("features", [0, 1, 2])
    assert nt.tensor.shape == (3, 256, 256, 3)

    # test index_select_tensor_dim
    t = nt_cat.index_select_tensor_dim("features", [0, 1, 2])
    assert t.shape == (3, 256, 256, 3)

    # test select_dim when returning NamedTensor
    with pytest.raises(ValueError):
        nt = nt_cat.select_dim("features", 0)
    nt = nt_cat.select_dim("lon", 0)
    assert nt.tensor.shape == (3, 256, 30)
    assert nt.feature_names == nt_cat.feature_names
    assert nt.names == ["batch", "lat", "features"]

    # test index_select_dim when returning NamedTensor
    nt = nt_cat.index_select_dim("features", [0, 1, 2])
    assert nt.tensor.shape == (3, 256, 256, 3)
    assert nt.feature_names == nt_cat.feature_names[:3]
    assert nt.names == ["batch", "lat", "lon", "features"]

    # test dim_size
    assert nt_cat.dim_size("features") == 30

    # test dim_index
    assert nt_cat.dim_index("features") == 3

    # test rearrange_
    nt_cat.rearrange_("batch lat lon features -> batch features lat lon")
    assert nt_cat.names == ["batch", "features", "lat", "lon"]
    assert nt_cat.tensor.shape == (3, 30, 256, 256)

    # test rearrange_ raises ValueError if wrong rearrangement
    with pytest.raises(ValueError):
        nt_cat.rearrange_("batch lat lon features -> batch features lat")

    # test rearrange_ raises ValueError if non existing dim is supplied
    with pytest.raises(ValueError):
        nt_cat.rearrange_("batch lat nodim features -> batch features lat lon")
