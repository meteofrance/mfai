from pathlib import Path
from typing import Optional, Tuple
from torch import nn
from .deeplabv3 import DeepLabV3, DeepLabV3Plus
from .half_unet import HalfUNet
from .segformer import Segformer
from .swinunetr import SwinUNETR
from .unet import UNet, CustomUnet
from .unetrpp import UNETRPP


all_nn_architectures = (
    DeepLabV3,
    DeepLabV3Plus,
    HalfUNet,
    Segformer,
    SwinUNETR,
    UNet,
    CustomUnet,
    UNETRPP,
)


def load_from_settings_file(
    model_name: str,
    in_channels: int,
    out_channels: int,
    settings_path: Path,
    input_shape: Optional[Tuple[int, ...]] = None,
) -> nn.Module:
    """
    Instanciate a model from a settings file with Schema validation.
    """

    # pick the class matching the supplied name
    model_kls = next(
        (kls for kls in all_nn_architectures if kls.__name__ == model_name), None
    )

    if model_kls is None:
        raise ValueError(
            f"Model {model_name} not found in available architectures: {[x.__name__ for x in all_nn_architectures]}"
        )

    # load the settings
    with open(settings_path, "r") as f:
        model_settings = model_kls.settings_kls.schema().loads(f.read())

    # instanciate the model
    return model_kls(
        in_channels, out_channels, input_shape=input_shape, settings=model_settings
    )
