import importlib
import pkgutil
from pathlib import Path
from typing import Optional, Tuple

from torch import nn
from .base import ModelABC


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
            f"Model {model_name} not found in available architectures: {[x for x in registry]}"
        )

    # load the settings
    with open(settings_path, "r") as f:
        model_settings = model_kls.settings_kls.schema().loads(f.read())

    # instanciate the model
    return model_kls(
        in_channels, out_channels, input_shape=input_shape, settings=model_settings
    )
