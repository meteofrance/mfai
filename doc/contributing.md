# Contributing

We welcome contributions to this package. Our guidelines are the following:

- Submit a PR with a clear description of the changes and the motivation behind them.
- Make sure the current tests pass and add new tests if necessary to cover the new features. Our CI will fail with a **test coverage below 80%**.
- Make sure the code is formatted with [ruff](https://docs.astral.sh/ruff/) : `ruff format` and `ruff check --select I --fix`
- Make sure the code respects our mypy type hinting requirements, see [the mypy default checks](https://mypy.readthedocs.io/en/stable/error_code_list.html#error-codes-enabled-by-default) and the [project's mypy configuration](https://github.com/meteofrance/mfai/blob/main/pyproject.toml).


## Contributing a new model

All models in mfai follow the same contract interface. You model should herit from both `pytorch.nn.Module` and `mfai.pytorch.models.base.ModelABC`. To ease the type checking, use the `mfai.pytorch.models.base.BaseModel` class. A settings class named `ModelNameSettings` should contain all of the models settings and be an instantiation parameter for the model class.

You can take example on an existing model implementation:
```py
# mfai/pytorch/models/half_unet.py

from dataclasses import dataclass
from dataclasses_json import dataclass_json
from mfai.pytorch.models.base import AutoPaddingModel, BaseModel, ModelType


@dataclass_json
@dataclass(slots=True)
class HalfUNetSettings:
    num_filters: int = 64
    dilation: int = 1
    bias: bool = False
    use_ghost: bool = False
    last_activation: str = "Identity"
    absolute_pos_embed: bool = False
    autopad_enabled: bool = False


class HalfUNet(BaseModel, AutoPaddingModel):
    settings_kls = HalfUNetSettings
    onnx_supported: bool = True
    supported_num_spatial_dims: tuple[int, ...] = (2,)
    num_spatial_dims: int = 2
    features_last: bool = False
    model_type: ModelType = ModelType.CONVOLUTIONAL
    register: bool = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_shape: tuple[int, int],
        settings: HalfUNetSettings = HalfUNetSettings(),
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_shape = input_shape
        self._settings = settings
```

## Contributing to the documentation

To generate the documentation locally, run the following commands from the root of the repository:

- `python doc/generate_rst.py`: generates the RST files for the API reference by introspecting the codebase
- `sphinx-apidoc -o doc/api/references mfai --force --templatedir doc/_templates/apidoc`: generates the full package reference using `sphinx-apidoc`
- `mv doc/api/references/modules.rst doc/api/references.rst`: moves the generated index to the expected location
- `make -C doc html`: builds the HTML documentation into `doc/_build/html/`

You can then open the documentation in your browser by opening `doc/_build/html/index.html` directly.

To get a live preview that automatically refreshes on file changes, you can use the [Live Preview](https://marketplace.visualstudio.com/items?itemName=ms-vscode.live-server) extension for VSCode.

Once installed, open `doc/_build/html/index.html` in VSCode and click **Show Preview** — the browser will automatically refresh every time you rebuild the documentation with `make -C doc html`.