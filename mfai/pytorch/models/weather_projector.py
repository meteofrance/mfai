"""
This model takes weather/2d inputs (batch, features, height, width)
and produces tokens for multimodal language models.
"""
from torch import nn, Tensor
from dataclasses import dataclass

import einops
from dataclasses_json import dataclass_json


class PatchMaker(nn.Module):
    """
    Converts a vision/weather (B, C, H, W) tensor into a (B, F, T) token tensor.
    Each token is built with all the data of one patch of size patch_size
    """
    def __init__(self, patch_size: int | tuple[int, int], input_dims: tuple[int, int], autopadding: bool = True):
        super().__init__()
        self.autopadding = autopadding
        
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        
        if autopadding:
            patch_x, patch_y = patch_size
            x_size, y_size = input_dims
            # checks if x requires padding
            x_padding = (patch_x - (x_size % patch_x)) % patch_x

            # checks if y requires padding
            y_padding = (patch_y - (y_size % patch_y)) % patch_y

            self.zero_pad = nn.ZeroPad2d([0, x_padding, 0, y_padding])

    def forward(self, t: Tensor) -> Tensor:
        """
        1. zero pad if padding is enabled
        2. check for dim consistency
        3. einops rearrange to patch
        """
        if self.autopadding:
            t = self.zero_pad(t)
        
        _, _, h, w = t.shape
        p1, p2 = self.patch_size

        if not h % p1 == 0 and w % p2 == 0:
            raise ValueError(f"input height {h} and width {w} MUST be multiples of patch_size {self.patch_size}")

        t = einops.rearrange(t, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p1, p2=p2)
        return t



@dataclass_json
@dataclass
class WeatherProjectorSettings:

    # patch_size
    patch_size: int = 8
   
    # lat, lon, timesteps, features
    input_dims: tuple[int, int, int] = (51, 51, 23)
    
    # target embedding dimension
    embedding_dim: int = 768


class WeatherProjector(nn.Module):
    def __init__(self, settings: WeatherProjectorSettings = WeatherProjectorSettings()):
        
        super().__init__()
        
        self.settings = settings
        this_dim = self.settings.patch_size*self.settings.patch_size*self.settings.input_dims[-1]
        
        self.patcher = PatchMaker(self.settings.patch_size, self.settings.input_dims[:2])
        self.proj = nn.Linear(this_dim, self.settings.embedding_dim)
        
    def forward(self, t: Tensor) -> Tensor:
        t = self.patcher(t)
        return self.proj(t)