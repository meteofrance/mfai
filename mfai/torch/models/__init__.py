from .deeplabv3 import DeepLabV3, DeepLabV3Plus
from .half_unet import HalfUnet
from .segformer import Segformer
from .swinunetr import SwinUNETR
from .unet import Unet
from .unetrpp import UNETRPP


all_nn_architectures = (
    DeepLabV3,
    DeepLabV3Plus,
    HalfUnet,
    Segformer,
    SwinUNETR,
    Unet,
    UNETRPP,
)
