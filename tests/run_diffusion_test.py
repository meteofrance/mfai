from test_models import train_model

from mfai.torch.models.gaussian_diffusion import GaussianDiffusion

train_model(GaussianDiffusion,(64,64))