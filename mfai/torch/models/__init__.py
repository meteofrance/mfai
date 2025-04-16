import importlib
import pkgutil
from pathlib import Path
from types import ModuleType


from torch import nn

from .base import AutoPaddingModel, ModelABC, ModelType

# Load all models from the torch.models package
# which are ModelABC subclasses and have the register attribute set to True
registry: dict[str, type[ModelABC] | type[AutoPaddingModel]] = dict()
package: ModuleType = importlib.import_module("mfai.torch.models")
for module_info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
    module: ModuleType = importlib.import_module(module_info.name)
    for object_name, kls in module.__dict__.items():
        if (
            isinstance(kls, type)
            and issubclass(kls, ModelABC)
            and kls != ModelABC
            and kls.register  # type: ignore[truthy-function]
        ):
            if kls.__name__ in registry:
                raise ValueError(
                    f"Model {kls.__name__} from plugin {object_name} already exists in the registry."
                )
            registry[kls.__name__] = kls
all_nn_architectures = list(registry.values())

nn_architectures: dict[ModelType, list[ModelABC]] = {
    model_type: [
        architecture for architecture in all_nn_architectures
        if architecture.model_type == model_type
    ]
    for model_type in ModelType
}

autopad_nn_architectures = {
    obj for obj in all_nn_architectures if issubclass(obj, AutoPaddingModel)
}


def load_from_settings_file(
    model_name: str,
    in_channels: int,
    out_channels: int,
    settings_path: Path,
    input_shape: tuple[int, ...],
) -> nn.Module:
    """
    Instanciate a model from a settings file with Schema validation.
    """

    # Pick the class matching the supplied name
    model_kls = registry.get(model_name, None)

    if model_kls is None:
        raise ValueError(
            f"Model {model_name} not found in available architectures: {[x for x in registry]}. Make sure the model's `registry` attribute is set to True (default is False)."
        )

    # Check that the class is ModelABC subclass
    if not (issubclass(model_kls, ModelABC) and issubclass(model_kls, nn.Module)):
        raise ValueError(
            f"Model {model_name} is not a subclass of mfai.torch.models.ModelABC and torch.nn.Module."
        )

    # Check that the model's settings class is wrapped by the @dataclass_json decorator by looking for the schema attribute
    if not hasattr(model_kls.settings_kls, "schema"):
        raise ValueError(
            f"Model {model_name}.settings_kls has no attribute schema. Make sure the model's settings class is wrapped by the @dataclass_json decorator."
        )

    # load the settings
    with open(settings_path, "r") as f:
        model_settings = model_kls.settings_kls.schema().loads(f.read())

    # instanciate the model
    return model_kls(
        in_channels, out_channels, input_shape=input_shape, settings=model_settings
    )
