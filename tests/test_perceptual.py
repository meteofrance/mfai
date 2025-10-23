import pytest
import torch

from mfai.pytorch.losses.perceptual import LPIPS, PerceptualLoss


def test_perceptual_loss_on_same_img() -> None:
    """
    Test of the Perceptual Loss on the same image
    """
    # Test with 3-channels images

    input = torch.rand(size=(1, 3, 224, 224))
    # Random VGG16
    perceptual_loss = PerceptualLoss(
        device="cpu",
        in_channels=3,
        multi_scale=False,
        channel_iterative_mode=False,
        pre_trained=False,
        resize_input=False,
    )
    loss = perceptual_loss.forward(input, input)
    assert pytest.approx(loss, 0.001) == 0

    # Random VGG16
    perceptual_loss = PerceptualLoss(
        device="cpu",
        in_channels=3,
        multi_scale=False,
        channel_iterative_mode=True,
        pre_trained=False,
        resize_input=False,
    )
    loss = perceptual_loss.forward(input, input)
    assert pytest.approx(loss, 0.001) == 0

    # Test with single channel images

    input = torch.rand(size=(1, 1, 224, 224))
    # Random VGG16
    perceptual_loss = PerceptualLoss(
        device="cpu",
        in_channels=1,
        multi_scale=False,
        channel_iterative_mode=False,
        pre_trained=False,
        resize_input=False,
    )
    loss = perceptual_loss.forward(input, input)
    assert pytest.approx(loss, 0.001) == 0

    # Random VGG16
    perceptual_loss = PerceptualLoss(
        device="cpu",
        in_channels=1,
        multi_scale=False,
        channel_iterative_mode=True,
        pre_trained=False,
        resize_input=False,
    )
    loss = perceptual_loss.forward(input, input)
    assert pytest.approx(loss, 0.001) == 0


def test_perceptual_loss_on_different_img() -> None:
    """
    Test of the Perceptual Loss on the different image
    """

    # Test with 3-channels images

    preds = torch.rand(size=(1, 3, 224, 224))
    input = torch.rand(size=(1, 3, 224, 224))
    # Random VGG16
    perceptual_loss = PerceptualLoss(
        device="cpu",
        in_channels=3,
        multi_scale=False,
        channel_iterative_mode=False,
        pre_trained=False,
        resize_input=False,
    )
    loss = perceptual_loss.forward(input, preds)
    assert pytest.approx(loss, 0.001) != 0

    # Random VGG16
    perceptual_loss = PerceptualLoss(
        device="cpu",
        in_channels=3,
        multi_scale=False,
        channel_iterative_mode=True,
        pre_trained=False,
        resize_input=False,
    )
    loss = perceptual_loss.forward(input, preds)
    assert pytest.approx(loss, 0.001) != 0

    # Test with  single channel images

    preds = torch.rand(size=(1, 1, 224, 224))
    input = torch.rand(size=(1, 1, 224, 224))
    # Random VGG16
    perceptual_loss = PerceptualLoss(
        device="cpu",
        in_channels=1,
        multi_scale=False,
        channel_iterative_mode=False,
        pre_trained=False,
        resize_input=False,
    )
    loss = perceptual_loss.forward(input, preds)
    assert pytest.approx(loss, 0.001) != 0

    # Random VGG16
    perceptual_loss = PerceptualLoss(
        device="cpu",
        in_channels=1,
        multi_scale=False,
        channel_iterative_mode=True,
        pre_trained=False,
        resize_input=False,
    )
    loss = perceptual_loss.forward(input, preds)
    assert pytest.approx(loss, 0.001) != 0


def test_feature_computation() -> None:
    """
    Test of the Perceptual Loss on feature computation
    """
    preds = torch.rand(size=(1, 1, 224, 224))
    # Random VGG16
    perceptual_loss = PerceptualLoss(
        device="cpu",
        multi_scale=False,
        in_channels=1,
        channel_iterative_mode=False,
        pre_trained=False,
        resize_input=False,
        feature_block_ids=[0],
        feature_layer_ids=[4],
        style_block_ids=[0],
        style_layer_ids=[4],
        alpha_feature=1,
        alpha_style=1,
    )

    features, styles = perceptual_loss.compute_perceptual_features(
        preds, return_features_and_styles=True
    )

    assert len(features) == 1
    assert len(styles) == 1
    assert features[0][0].shape == (1, 64, 224, 224)

    # Test multi_scale
    # Random VGG16
    perceptual_loss = PerceptualLoss(
        device="cpu",
        multi_scale=True,
        in_channels=1,
        channel_iterative_mode=False,
        pre_trained=False,
        resize_input=True,
        feature_block_ids=[0],
        feature_layer_ids=[4],
        style_block_ids=[0],
        style_layer_ids=[4],
        alpha_feature=1,
        alpha_style=1,
    )

    features, styles = perceptual_loss.compute_perceptual_features(
        preds, return_features_and_styles=True
    )

    assert len(features) == 4
    assert len(styles) == 4
    assert all(
        [features[i][0].shape == (1, 64, 224, 224) for i in range(len(features))]
    )
    assert all([styles[i][0].shape == (1, 64, 64) for i in range(len(styles))])

    # Test multi blocks
    # Random VGG16
    perceptual_loss = PerceptualLoss(
        device="cpu",
        multi_scale=False,
        in_channels=1,
        channel_iterative_mode=False,
        pre_trained=False,
        resize_input=False,
        feature_block_ids=[0, 1],
        feature_layer_ids=[4, 9],
        style_block_ids=[0, 1],
        style_layer_ids=[4, 9],
        alpha_feature=1,
        alpha_style=1,
    )

    features, styles = perceptual_loss.compute_perceptual_features(
        preds, return_features_and_styles=True
    )

    assert len(features) == 1
    assert len(styles) == 1
    assert len(features[0]) == 2
    assert len(styles[0]) == 2
    assert features[0][0].shape == (1, 64, 224, 224)
    assert features[0][1].shape == (1, 128, 112, 112)
    assert styles[0][0].shape == (1, 64, 64)
    assert styles[0][1].shape == (1, 128, 128)


def test_lpips_on_same_img() -> None:
    """
    Test of the Perceptual Loss on the same image
    """
    # Test with 3-channel images

    input = torch.rand(size=(1, 3, 224, 224))
    # Random VGG16
    lpips = LPIPS(
        device="cpu",
        in_channels=3,
        multi_scale=False,
        channel_iterative_mode=False,
        pre_trained=False,
        resize_input=False,
    )
    loss = lpips.forward(input, input)
    assert pytest.approx(loss, 0.001) == 0

    # Random VGG16
    lpips = LPIPS(
        device="cpu",
        in_channels=3,
        multi_scale=False,
        channel_iterative_mode=True,
        pre_trained=False,
        resize_input=False,
    )
    loss = lpips.forward(input, input)
    assert pytest.approx(loss, 0.001) == 0

    # Test with single channel images

    input = torch.rand(size=(1, 1, 224, 224))
    # Random VGG16
    lpips = LPIPS(
        device="cpu",
        in_channels=1,
        multi_scale=False,
        channel_iterative_mode=False,
        pre_trained=False,
        resize_input=False,
    )
    loss = lpips.forward(input, input)
    assert pytest.approx(loss, 0.001) == 0

    # Random VGG16
    lpips = LPIPS(
        device="cpu",
        in_channels=1,
        multi_scale=False,
        channel_iterative_mode=True,
        pre_trained=False,
        resize_input=False,
    )
    loss = lpips.forward(input, input)
    assert pytest.approx(loss, 0.001) == 0


def test_lpips_on_different_img() -> None:
    """
    Test of the Perceptual Loss on the different image
    """

    # Test with 3-channel images

    preds = torch.rand(size=(1, 3, 224, 224))
    input = torch.rand(size=(1, 3, 224, 224))
    # Random VGG16
    lpips = LPIPS(
        device="cpu",
        in_channels=3,
        multi_scale=False,
        channel_iterative_mode=False,
        pre_trained=False,
        resize_input=False,
    )
    loss = lpips.forward(input, preds)
    assert pytest.approx(loss, 0.001) != 0

    # Random VGG16
    lpips = LPIPS(
        device="cpu",
        in_channels=3,
        multi_scale=False,
        channel_iterative_mode=True,
        pre_trained=False,
        resize_input=False,
    )
    loss = lpips.forward(input, preds)
    assert pytest.approx(loss, 0.001) != 0

    # Test with single channel images

    preds = torch.rand(size=(1, 1, 224, 224))
    input = torch.rand(size=(1, 1, 224, 224))
    # Random VGG16
    lpips = LPIPS(
        device="cpu",
        in_channels=1,
        multi_scale=False,
        channel_iterative_mode=False,
        pre_trained=False,
        resize_input=False,
    )
    loss = lpips.forward(input, preds)
    assert pytest.approx(loss, 0.001) != 0

    # Random VGG16
    lpips = LPIPS(
        device="cpu",
        in_channels=1,
        multi_scale=False,
        channel_iterative_mode=True,
        pre_trained=False,
        resize_input=False,
    )
    loss = lpips.forward(input, preds)
    assert pytest.approx(loss, 0.001) != 0
