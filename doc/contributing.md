# Contributing

We welcome contributions to this package. Our guidelines are the following:

- Submit a PR with a clear description of the changes and the motivation behind them.
- Make sure the current tests pass and add new tests if necessary to cover the new features. Our CI will fail with a **test coverage below 80%**.
- Make sure the code is formatted with [ruff](https://docs.astral.sh/ruff/) : `ruff format` and `ruff check --select I --fix`
- Make sure the code respects our mypy type hinting requirements, see [the mypy default checks](https://mypy.readthedocs.io/en/stable/error_code_list.html#error-codes-enabled-by-default) and the [project's mypy configuration](https://github.com/meteofrance/mfai/blob/main/pyproject.toml).


## Cloning the project

```
git clone https://github.com/meteofrance/mfai
cd mfai
python3.10 -m venv .venv
source .venv/bin/activate
pip install .[dev, llm]
```

## Contributing a new model

You model should implement mfai's models contract interface described here.

Except for LLMs and MLLMs, each model we provide is a subclass of [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) and can be used in a PyTorch training loop. It has multiple class attributes to facilitate model usage in a project:
- **settings_kls**: a class that defines the settings of the model (number of filters, kernel size, ...). It is used to instanciate the model with a specific configuration.
- **onnx_supported**: a boolean that indicates if the model can be exported to onnx. Our CI validates that the model can be exported to onnx and reloaded for inference.
- **supported_num_spatial_dims**: a tuple that describes the spatial dimensions of the input tensor supported by the model. A model that supports 2D spatial data will have **(2,)** as value. A model that supports 2d or 3d spatial data will have **(2, 3)** as value.
- **num_spatial_dims**: an integer that describes the number of spatial dimensions of the input/output tensor expected by the instance of the model, must be a value in **supported_num_spatial_dims**.
- **settings**: a runtime property returns the settings instance used to instanciate the model.
- **model_type**: an Enum describing the type of model: CONVOLUTIONAL, VISION_TRANSFORMER, GRAPH, LLM, MLLM.
- **features_last**: a boolean that indicates if the features dimension is the last dimension of the input/output tensor. If False, the features dimension is the second dimension of the input/output tensor.
- **register**: a boolean that indicates if the model should be registered in the **MODELS** registry. By default, it is set to False which allows the creation of intermediate subclasses not meant for direct use.

The Python interface contract for our model is enforced using [Python ABC](https://docs.python.org/3/library/abc.html) and in our case [ModelABC](https://github.com/meteofrance/mfai/blob/main/mfai/pytorch/models/base.py#L1) class. This class is combined to `torch.nn.Module` in [BaseModel](mfai/pytorch/models/base.py#L1).

```python
@dataclass_json
@dataclass(slots=True)
class HalfUNetSettings:
    num_filters: int = 64
    dilation: int = 1
    bias: bool = False
    use_ghost: bool = False
    last_activation: str = "Identity"
    absolute_pos_embed: bool = False

class HalfUNet(BaseModel):
    settings_kls = HalfUNetSettings
    onnx_supported: bool = True
    supported_num_spatial_dims = (2,)
    num_spatial_dims: int = 2
    features_last: bool = False
    model_type: int = ModelType.CONVOLUTIONAL
    register: bool = True
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


## Write a unit test

If you contribute a model, and it correctly implement its contract interface, then it will be already tested. For any other contribution, we expect a unit test to be written in the `tests/` directory.


## Testing your contribution

To check that your code will pass the project's CI pipelines, you should run:
```sh
ruff format
ruff check --fix
mypy
pytest tests/
```

If all those commands succeed, you can move onto creating a Pull Request.

## Creating a pull request

We ask you to:
- Choose an explicit title for your pull request.
- Write a small paragraph explaining what changes are introduced to the project.
- Keep your pull request as draft until it passes the CI checks.
- Assign yourself to the pull request.
- Assign one or more maintainer as reviewers.