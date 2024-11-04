import pytest
import torch

from mfai.torch.namedtensor import NamedTensor


def test_named_tensor():
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
    )
    assert nt4.spatial_dim_idx == [1, 2]
    with pytest.raises(ValueError):
        nt | nt4

    # different number of dims => ValueError
    nt5 = NamedTensor(
        torch.rand(3, 256, 50),
        names=["batch", "lat", "lon"],
        feature_names=[f"feature_{i}" for i in range(50)],
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
        torch.rand(3, 256),
        names=["batch", "features"],
        feature_names=[f"f_{i}" for i in range(256)],
    )
    nt6.unsqueeze_and_expand_from_(nt)
    assert nt6.tensor.shape == (3, 256, 256, 256)
    assert nt6.names == ["batch", "lat", "lon", "features"]

    # test flattening lat,lon to ndims to simulate gridded data with 2D spatial dims into a GNN
    nt6.flatten_("ngrid", 1, 2)
    assert nt6.tensor.shape == (3, 65536, 256)
    assert nt6.names == ["batch", "ngrid", "features"]
    assert nt6.spatial_dim_idx == [1]

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
    nt_collate = NamedTensor.collate([nt10, nt11, nt12])
    assert nt_collate.tensor.shape == (3, 3, 256, 256, 10)
    assert nt_collate.feature_names == [f"feature_{i}" for i in range(10)]
    assert nt_collate.names == ["batch", "timesteps", "lat", "lon", "features"]