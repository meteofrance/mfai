import importlib
import pkgutil
from pathlib import Path
from typing import Optional, Tuple

import torch.utils.model_zoo as model_zoo
from torch import nn
from torchvision.models.resnet import BasicBlock, Bottleneck

from mfai.torch.models.resnet import ResNetEncoder

from .base import AutoPaddingModel, ModelABC

# Load all models from the torch.models package
# which are ModelABC subclasses and have the register attribute set to True
registry = dict()
package = importlib.import_module("mfai.torch.models")
for _, name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
    module = importlib.import_module(name)
    for object_name, kls in module.__dict__.items():
        if (
            isinstance(kls, type)
            and issubclass(kls, ModelABC)
            and kls != ModelABC
            and kls.register
        ):
            if kls.__name__ in registry:
                raise ValueError(
                    f"Model {kls.__name__} from plugin {object_name} already exists in the registry."
                )
            registry[kls.__name__] = kls
all_nn_architectures = list(registry.values())


autopad_nn_architectures = {
    obj
    for obj in all_nn_architectures
    if issubclass(obj, AutoPaddingModel) and obj != "AutoPaddingModel"
}


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
    model_kls = registry.get(model_name, None)

    if model_kls is None:
        raise ValueError(
            f"Model {model_name} not found in available architectures: {[x for x in registry]}. Make sure the model's `registry` attribute is set to True (default is False)."
        )

    # load the settings
    with open(settings_path, "r") as f:
        model_settings = model_kls.settings_kls.schema().loads(f.read())

    # instanciate the model
    return model_kls(
        in_channels, out_channels, input_shape=input_shape, settings=model_settings
    )


##########################################################################################################
######################################         Encoders           ########################################
##########################################################################################################


ENCODERS_MAP = {
    "resnet18": {
        "encoder": ResNetEncoder,
        "pretrained_url": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet18-d92f0530.pth",  # noqa
        "params": {
            "out_channels": (3, 64, 64, 128, 256, 512),
            "block": BasicBlock,
            "layers": [2, 2, 2, 2],
        },
    },
    "resnet34": {
        "encoder": ResNetEncoder,
        "pretrained_url": "https://download.pytorch.org/models/resnet34-b627a593.pth",
        "params": {
            "out_channels": (3, 64, 64, 128, 256, 512),
            "block": BasicBlock,
            "layers": [3, 4, 6, 3],
        },
    },
    "resnet50": {
        "encoder": ResNetEncoder,
        "pretrained_url": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet50-08389792.pth",  # noqa
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 6, 3],
        },
    },
}


def get_vision_encoder(
    name: str,
    in_channels: int = 3,
    depth: int = 5,
    weights: bool = True,
    output_stride: int = 32,
):
    """
    Return an encoder with pretrained weights or not.
    """
    try:
        Encoder = ENCODERS_MAP[name]["encoder"]
    except KeyError:
        raise KeyError(
            "Wrong encoder name `{}`, supported ENCODERS_MAP: {}".format(
                name, list(ENCODERS_MAP.keys())
            )
        )

    params = ENCODERS_MAP[name]["params"]
    params.update(depth=depth)
    encoder = Encoder(**params)

    if weights:
        url = ENCODERS_MAP[name]["pretrained_url"]
        pretrained = True
        if url is None:
            pretrained = False
            raise KeyError(
                f"No url is available for the pretrained encoder choosen ({name})."
            )
        encoder.load_state_dict(model_zoo.load_url(url))
    else:
        pretrained = False

    encoder.set_in_channels(in_channels, pretrained=pretrained)
    if output_stride != 32:
        encoder.make_dilated(output_stride)

    return encoder
