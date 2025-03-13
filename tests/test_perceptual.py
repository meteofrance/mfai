import pytest
import torch

from mfai.torch.losses.perceptual import PerceptualLoss


def test_perceptual_loss_on_same_img():
    """
    Test of the Perceptual Loss on the same image
    """
    input = torch.rand(size=(1,3,224,224))
    # Random VGG16 
    perceptual_loss = PerceptualLoss(
        device='cpu',
        multi_scale=False,
        channel_iterative_mode=False,
        pre_trained=False,
        resize_input=False
    )
    loss = perceptual_loss.forward(input, input)
    assert pytest.approx(loss, 0.001) == 0


def test_perceptual_loss_on_different_img():
    """
    Test of the Perceptual Loss on the same image
    """
    preds = torch.rand(size=(1,3,224,224))
    input = torch.rand(size=(1,3,224,224))
    # Random VGG16 
    perceptual_loss = PerceptualLoss(
        device='cpu',
        multi_scale=False,
        channel_iterative_mode=False,
        pre_trained=False,
        resize_input=False
    )
    loss = perceptual_loss.forward(input, preds)
    assert pytest.approx(loss, 0.001) != 0

def test_feature_computation():
    """
    Test of the Perceptual Loss on the same image
    """
    preds = torch.rand(size=(1,1,224,224))
    # Random VGG16 
    perceptual_loss = PerceptualLoss(
        device='cpu',
        multi_scale=False,
        channel_iterative_mode=False,
        pre_trained=False,
        resize_input=False
    )
    
    features, styles = perceptual_loss.compute_perceptual_features(preds, return_features_and_styles=True)
    assert len(features) == 1
    assert len(styles) == 1
