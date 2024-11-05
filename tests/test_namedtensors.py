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
    nt_collate = NamedTensor.collate_fn([nt10, nt11, nt12])
    assert nt_collate.tensor.shape == (3, 3, 256, 256, 10)
    assert nt_collate.feature_names == [f"feature_{i}" for i in range(10)]
    assert nt_collate.names == ["batch", "timesteps", "lat", "lon", "features"]

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

    # Test iteration on batch dimension with bare_tensor=False
    for i, nt_dim in enumerate(nt.iter_dim("batch", bare_tensor=False)):
        assert nt_dim.tensor.shape == (256, 256, 50)
        assert nt_dim.names == ["lat", "lon", "features"]

    assert i == 2

    # Test iteration on batch dimension with bare_tensor=True
    for i, tensor in enumerate(nt.iter_dim("batch", bare_tensor=True)):
        assert tensor.shape == (256, 256, 50)

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

    # test select_dim along the features dim
    t = nt_cat.select_dim("features", 0)
    assert t.shape == (3, 256, 256)

    # test select_dim along the lat dim
    t = nt_cat.select_dim("lat", 128)
    assert t.shape == (3, 256, 30)

    # test index_select_dim
    t = nt_cat.index_select_dim("features", [0, 1, 2])
    assert t.shape == (3, 256, 256, 3)

    # test select_dim when returning NamedTensor
    with pytest.raises(ValueError):
        t = nt_cat.select_dim("features", 0, bare_tensor=False)
    t = nt_cat.select_dim("lon", 0, bare_tensor=False)
    assert t.tensor.shape == (3, 256, 30)
    assert t.feature_names == nt_cat.feature_names
    assert t.names == ["batch", "lat", "features"]

    # test index_select_dim when returning NamedTensor
    t = nt_cat.index_select_dim("features", [0, 1, 2], bare_tensor=False)
    assert t.tensor.shape == (3, 256, 256, 3)
    assert t.feature_names == nt_cat.feature_names[:3]
    assert t.names == ["batch", "lat", "lon", "features"]

    # test dim_size
    assert nt_cat.dim_size("features") == 30

    # test dim_index
    assert nt_cat.dim_index("features") == 3
