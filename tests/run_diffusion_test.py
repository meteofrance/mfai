from test_models import train_model

from mfai.torch.models.gaussian_diffusion import GaussianDiffusion

in_channels = 1
out_channels = 1
input_shape = (64, 64)

train_model(
    GaussianDiffusion(
        in_channels=in_channels, out_channels=out_channels, input_shape=input_shape
    ),
    (in_channels, 64, 64),
)
