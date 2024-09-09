# MFAI: Météo-France's AI Python package

![Unit Tests](https://github.com/meteofrance/mfai/actions/workflows/tests.yml/badge.svg)

**MFAI** is a Python package that provides the following features:
- A variety of PyTorch Neural Network architectures (CNN, Vision Transformers, ...) adapted to our needs, tested on our projects and datasets. For each architecture, we provide the reference to the original paper and source code if applicable and also the modifications we made.
- Per architecture schema validated settings using [dataclasses-json](https://github.com/lidatong/dataclasses-json)
- A NamedTensor class to handle multi-dimensional data with named dimensions and named features

# Table of contents

- [Neural Network Architectures](#neural-network-architectures)
    - deeplabv3/deeplabv3+
    - halfunet
    - unet/customunet
    - segformer
    - swinunetr
    - unetr++
- [SegmentationLightningModule](#segmentationlightningmodule)
- [NamedTensors](#namedtensors)
- [Metrics](#metrics)
    - Critical Sucess Index
    - False Alarm Rate
    - False Negative Rate
    - Precision-Recall Area Under Curve
- [Installation](#installation)
- [Usage](#usage)
    - [Instanciate a model](#instanciate-a-model)
    - [Export to onnx](#export-to-onnx)
    - [NamedTensors](#namedtensors-example)
- [Running tests](#tests)

# Neural Network Architectures

Each model we provide is a subclass of [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) and can be used in a PyTorch training loop. It has three **critical** class attributes:
- **settings_kls**: a class that defines the settings of the model (number of filters, kernel size, ...). It is used to instanciate the model with a specific configuration.
- **onnx_supported**: a boolean that indicates if the model can be exported to onnx. Our CI validates that the model can be exported to onnx and reloaded for inference.
- **input_spatial_dims**: a tuple that describes the spatial dimensions of the input tensor supported by the model. A model that supports 2D spatial data will have **(2,)** as value. A model that supports 2d or 3d spatial data will have **(2, 3)** as value.

The Python interface contract for our model is enforced using [Python ABC](https://docs.python.org/3/library/abc.html) and in our case [ModelABC](mfai/torch/models/base.py#L1) class.

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

class HalfUNet(ModelABC, nn.Module):
    settings_kls = HalfUNetSettings
    onnx_supported = True
    input_spatial_dims = (2,)
```

Currently we support the following neural network architectures:

| Model  | Research Paper  | Input Shape    | ONNX exportable ? | Notes | Use-Cases at MF | Maintainer(s) |
| :---:   | :---: | :---: | :---: | :---: | :---: | :---: |
| [DeepLabV3Plus](mfai/torch/models/deeplabv3.py#L1) | [arxiv link](https://arxiv.org/abs/1802.02611) | (Batch, features, Height, Width)    | Yes | As a very large receptive field versus U-Net, Half-Unet, ... | Front Detection, Nowcasting | Theo Tournier / Frank Guibert |
| [HalfUNet](mfai/torch/models/half_unet.py#L1) | [researchgate link](https://www.researchgate.net/publication/361186968_Half-UNet_A_Simplified_U-Net_Architecture_for_Medical_Image_Segmentation) | (Batch, features, Height, Width)    | Yes | In prod/oper on [Espresso](https://www.mdpi.com/2674-0494/2/4/25) V2 with 128 filters and standard conv blocks instead of ghost | Satellite channels to rain estimation |  Frank Guibert |
| [UNet](mfai/torch/models/unet.py#L1) | [arxiv link](https://arxiv.org/pdf/1505.04597.pdf) | (Batch, features, Height, Width)    | Yes | Vanilla U-Net | Radar image cleaning |  Theo Tournier / Frank Guibert |
| [CustomUnet](mfai/torch/models/unet.py#L1) | [arxiv link](https://arxiv.org/pdf/1505.04597.pdf) | (Batch, features, Height, Width)    | Yes | U-Net like architecture with a variety of resnet encoder choices | Radar image cleaning |  Theo Tournier |
| [Segformer](mfai/torch/models/segformer.py#L1) | [arxiv link](https://arxiv.org/abs/2105.15203)   | (Batch, features, Height, Width) | Yes | On par with u-net like on Deepsyg (MF internal), added an upsampling stage. Adapted from [Lucidrains' github](https://github.com/lucidrains/segformer-pytorch) | Segmentation tasks | Frank Guibert |
| [SwinUNETR](mfai/torch/models/swinunetr.py#L1) | [arxiv link](https://arxiv.org/abs/2201.01266)   | (Batch, features, Height, Width)  | No | 2D Swin  Unet transformer (Pangu and archweather uses customised 3D versions of Swin Transformers). Plugged in from [MONAI](https://github.com/Project-MONAI/MONAI/). The decoders have been modified to use Bilinear2D + Conv2d instead of Conv2dTranspose to remove artefacts/checkerboard effects | Segmentation tasks  |  Frank Guibert |
| [UNETRPP](mfai/torch/models/unetrpp.py#L1) | [arxiv link](https://arxiv.org/abs/2212.04497)  | (Batch, features, Height, Width) or  (Batch, features, Height, Width, Depth) | Yes | Vision transformer with a reduced GFLOPS footprint adapted from [author's github](https://github.com/Amshaker/unetr_plus_plus). Modified to work both with 2d and 3d inputs | Front Detection | Frank Guibert |

# SegmentationLightningModule

We provide **SegmentationLightningModule** a lightning module adapted to supervised Deep Learning projects where the input of the neural network is made of one or multiple images and the target is also one or multiple images.

The module can be instanciated with any of the aforementioned neural networks architetures and used in 4 different modes : binary classification, multiclass classification, multilabel classification and regression.

The module provides:
- customization for each stage of the training
- metrics and plot logging with tensorboard
- logging of configuration and hyperparameters
- computation of several metrics during validation stage
- test stage: compute metrics for each sample individualy and save them in CSV file

# Lightning CLI

Lightning CLI is a convinient way to easily configure your DL experiments and reduce the number of lines of code in your project.

We provide an example of usage of the Lightning CLI with our lightning module and an exemple of config file to launch an experiment.

# NamedTensors

PyTorch provides an experimental feature called [**named tensors**](https://pytorch.org/docs/stable/named_tensor.html), at this time it is subject to change so we don't use it. That's why we provide our own implementation.

NamedTensors are a way to give names to dimensions of tensors and to keep track of the names of the physical/weather parameters along the features dimension.

The [**NamedTensor**](../py4cast/datasets/base.py#L38) class is a wrapper around a PyTorch tensor, it allows us to pass consistent object linking data and metadata with extra utility methods (concat along features dimension, flatten in place, ...). See the implementation [here](../py4cast/datasets/base.py#L38) and usage for plots [here](../py4cast/observer.py)

# Metrics

In addition to metrics available in [**torchmetrics**](https://lightning.ai/docs/torchmetrics/stable/), we implement :
- Criticall Sucess Index (CSI) is given by: TP / (TP+FP+FN). This metric, usefull in meteorology, takes into account both false alarms and missed events in a neighborhood to avoid the phenomenon of double penalty.
- False Alarm Rate (FAR) is given by: FP / (FP + TP).
- False Negative Rate (FNR) is given by: FN / (FN + TP).
- Precision-Recall Area Under the Curve (PR AUC). This metric summarize the overall performance of a model without depending on a threshold. It can be used in place of the Area Under ROC Curve when the dataset is too unbalanced.

# Installation

We will soon push the package to PyPI. In the meantime, you can install it from the source code.

## Cloning the repository

```bash
git clone https://github.com/meteofrance/mfai
cd mfai
pip install -e .
```

# Usage

## Instanciate a model

Our [unit tests](tests/test_models.py#L39) provides an example of how to use the models in a PyTorch training loop. Our models are instanciated with 2 mandatory positional arguments: **in_channels** and **out_channels** respectively the number of input and output channels/features of the model. A third **input_shape** parameter is either mandatory (**UNETR++** or **HalfUNet with absolute pos embedding**) or optional for the other models. It describes the shape of the input tensor along its spatial dimensions.

The last parameter is an instance of the model's settings class and is a keyword argument with a default value set to the default settings.



Here is an example of how to instanciate the UNet model with a 3 channels input (like an RGB image) and 1 channel output with its default settings:

```python
from mfai.torch.models import UNet
unet = UNet(in_channels=3, out_channels=1)
```

**_FEATURE:_** Once instanciated, the model (subclass of **nn.Module**) can be used like any standard [PyTorch model](https://pytorch.org/tutorials/beginner/introyt/trainingyt.html).

In order to instanciate a HalfUNet model with a 2 channels inputs, 2 channels outputs and a custom settings (128 filters, ghost module):

```python
from mfai.torch.models import HalfUNet
halfunet = HalfUNet(in_channels=2, out_channels=2, settings=HalfUNet.settings_kls(num_filters=128, use_ghost=True))
```

Finally, to instanciate a model with the mandatory **input_shape** parameter, here is an example with the UNETR++ model working on 2d spatial data (256x256) with 3 channels input and 1 channel output:

```python
from mfai.torch.models import UNETRPP
unetrpp = UNETRPP(in_channels=3, out_channels=1, input_shape=(256, 256))
```

**_FEATURE:_**  Each model has its settings class available under the **settings_kls** attribute.

You can use the **load_from_settings_file** function to instanciate a model with its settings from a json file:

```python
from pathlib import Path
from mfai.torch.models import load_from_settings_file
model = load_from_settings_file(
    "HalfUNet",
    2,
    2,
    Path(".") / "mfai" / "config" / "models" / "halfunet128.json",
)
```

**_FEATURE:_**  Use the **load_from_settings_file** to have the strictest validation of the settings.

## Export to onnx

Our tests [illustrate how to export and later reload a model to/from onnx](tests/test_models.py#L91). Here is an example of how to export a model to onnx:

```python
from mfai.torch import export_to_onnx, onnx_load_and_infer

# Export the model to onnx assuming we have just trained a 'model'
export_to_onnx(model, "model.onnx")

# Load the onnx model and infer
output_tensor = onnx_load_and_infer("model.onnx", input_tensor)
```

Check the code of [onnx_load_and_infer](mfai/torch/__init__.py#L35) if you would like to load the model once and make multiple inferences.


## SegmentationLightningModule

The lightning module can be instantiated and used in a forward pass as follows:

```python
arch = UNet(in_channels=1, out_channels=1, input_shape=[64, 64])
loss = torch.nn.MSELoss()
model = SegmentationLightningModule(arch, "binary", loss)
x = torch.randn((1, 1, 64, 64)).float()
model(x)
```

To train the model:
```python
from mfai.torch.dummy_dataset import DummyDataset

dataset = DummyDataset("train")
dataloader = DataLoader(dataset)
arch = UNet(in_channels=1, out_channels=1, input_shape=[64, 64])
loss = torch.nn.MSELoss()
model = SegmentationLightningModule(arch, "binary", loss)
trainer = L.Trainer()
trainer.fit(model=model, train_dataloaders=dataloader)
```

See [pytorch lightning documentation](https://lightning.ai/docs/overview/getting-started) for how to configure the Trainer and customize the module to suit your needs.

## Lightning CLI

Setting up lightning CLI is as easy as our `main.py` script:

```python
from lightning.pytorch.cli import LightningCLI

from mfai.torch.dummy_dataset import DummyDataModule
from mfai.torch.segmentation_module import SegmentationLightningModule


def cli_main():
    cli = LightningCLI(SegmentationLightningModule, DummyDataModule)  # noqa: F841


if __name__ == "__main__":
    cli_main()
```

Then launch your experiment with:
`python main.py {fit, validate, test, predict} YOUR_MODEL_AND_TRAINER_ARGUMENTS`

For instance:
`python main.py fit --model.model=Segformer --model.type_segmentation=binary --model.loss=torch.nn.BCEWithLogitsLoss --model.model.in_channels=2 --model.model.out_channels=1 --model.model.input_shape=[64, 64] --optimizer=AdamW --trainer.fast_dev_run=True`

`python main.py test --ckpt_path logs/best.ckpt`

To avoid very very long command lines, you can use a config file to setup your experiment:

For instance, see `mfai/config/cli_fit_test.yaml`:
```yaml
seed_everything: true
model:
  model:
    class_path: mfai.torch.models.Segformer
    init_args:
      in_channels: 2
      out_channels: 1
      input_shape: [64, 64]
      settings:
        num_layers: 2
        decoder_dim: 256
        num_downsampling_chans: 32
  type_segmentation: "binary"
  loss: torch.nn.BCEWithLogitsLoss
data:
  batch_size: 4
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: 0.001
...
```

Then you can pass your config file as argument to the CLI:
`python main.py fit --config mfai/config/cli_fit_test.yaml`

You can override arguments of the config file or add new ones in the CLI:
`python main.py fit --config mfai/config/cli_fit_test.yaml --optimizer.lr 0.0001 --trainer.fast_dev_run True`

------------

Don't be shy about using the CLI help tool! It can save you a lot of time:

<details>
<summary>runai python main.py fit --help (click to expand)</summary>

```bash
usage: main.py [options] fit [-h] [-c CONFIG] [--print_config[=flags]] [--seed_everything SEED_EVERYTHING] [--trainer CONFIG]
                             [--trainer.accelerator.help CLASS_PATH_OR_NAME] [--trainer.accelerator ACCELERATOR]
                             [--trainer.strategy.help CLASS_PATH_OR_NAME] [--trainer.strategy STRATEGY] [--trainer.devices DEVICES]
                             [--trainer.num_nodes NUM_NODES] [--trainer.precision PRECISION]
                             [--trainer.logger.help CLASS_PATH_OR_NAME] [--trainer.logger LOGGER]
                             [--trainer.callbacks.help CLASS_PATH_OR_NAME] [--trainer.callbacks CALLBACKS]
                             [--trainer.fast_dev_run FAST_DEV_RUN] [--trainer.max_epochs MAX_EPOCHS]
                             [--trainer.min_epochs MIN_EPOCHS] [--trainer.max_steps MAX_STEPS] [--trainer.min_steps MIN_STEPS]
                             [--trainer.max_time MAX_TIME] [--trainer.limit_train_batches LIMIT_TRAIN_BATCHES]
                             [--trainer.limit_val_batches LIMIT_VAL_BATCHES] [--trainer.limit_test_batches LIMIT_TEST_BATCHES]
                             [--trainer.limit_predict_batches LIMIT_PREDICT_BATCHES] [--trainer.overfit_batches OVERFIT_BATCHES]
                             [--trainer.val_check_interval VAL_CHECK_INTERVAL]
                             [--trainer.check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH]
                             [--trainer.num_sanity_val_steps NUM_SANITY_VAL_STEPS] [--trainer.log_every_n_steps LOG_EVERY_N_STEPS]
                             [--trainer.enable_checkpointing {true,false,null}] [--trainer.enable_progress_bar {true,false,null}]
                             [--trainer.enable_model_summary {true,false,null}]
                             [--trainer.accumulate_grad_batches ACCUMULATE_GRAD_BATCHES]
                             [--trainer.gradient_clip_val GRADIENT_CLIP_VAL]
                             [--trainer.gradient_clip_algorithm GRADIENT_CLIP_ALGORITHM] [--trainer.deterministic DETERMINISTIC]
                             [--trainer.benchmark {true,false,null}] [--trainer.inference_mode {true,false}]
                             [--trainer.use_distributed_sampler {true,false}] [--trainer.profiler.help CLASS_PATH_OR_NAME]
                             [--trainer.profiler PROFILER] [--trainer.detect_anomaly {true,false}]
                             [--trainer.barebones {true,false}] [--trainer.plugins.help CLASS_PATH_OR_NAME]
                             [--trainer.plugins PLUGINS] [--trainer.sync_batchnorm {true,false}]
                             [--trainer.reload_dataloaders_every_n_epochs RELOAD_DATALOADERS_EVERY_N_EPOCHS]
                             [--trainer.default_root_dir DEFAULT_ROOT_DIR] [--model CONFIG] [--model.model.help CLASS_PATH_OR_NAME]
                             --model.model MODEL --model.type_segmentation {binary,multiclass,multilabel,regression}
                             --model.loss LOSS [--data CONFIG] [--data.batch_size BATCH_SIZE]
                             [--optimizer.help CLASS_PATH_OR_NAME] [--optimizer CONFIG | CLASS_PATH_OR_NAME | .INIT_ARG_NAME VALUE]
                             [--lr_scheduler.help CLASS_PATH_OR_NAME]
                             [--lr_scheduler CONFIG | CLASS_PATH_OR_NAME | .INIT_ARG_NAME VALUE] [--ckpt_path CKPT_PATH]

Runs the full optimization routine.

options:
  -h, --help            Show this help message and exit.
  -c CONFIG, --config CONFIG
                        Path to a configuration file in json or yaml format.
  --print_config[=flags]
                        Print the configuration after applying all other arguments and exit. The optional flags customizes the
                        output and are one or more keywords separated by comma. The supported flags are: comments, skip_default,
                        skip_null.
  --seed_everything SEED_EVERYTHING
                        Set to an int to run seed_everything with this value before classes instantiation.Set to True to use a
                        random seed. (type: Union[bool, int], default: True)

Customize every aspect of training via flags:
  --trainer CONFIG      Path to a configuration file.
  --trainer.accelerator.help CLASS_PATH_OR_NAME
                        Show the help for the given subclass of Accelerator and exit.
  --trainer.accelerator ACCELERATOR
                        Supports passing different accelerator types ("cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto") as well as
                        custom accelerator instances. (type: Union[str, Accelerator], default: auto, known subclasses:
                        lightning.pytorch.accelerators.CPUAccelerator, lightning.pytorch.accelerators.CUDAAccelerator,
                        lightning.pytorch.accelerators.MPSAccelerator, lightning.pytorch.accelerators.XLAAccelerator)
  --trainer.strategy.help CLASS_PATH_OR_NAME
                        Show the help for the given subclass of Strategy and exit.
```
</details>


<details>
<summary>runai python main.py fit  --model.model.help mfai.torch.models.Segformer (click to expand)</summary>

```bash
usage: main.py --model.model.in_channels IN_CHANNELS --model.model.out_channels OUT_CHANNELS
               --model.model.input_shape [ITEM,...] [--model.model.settings.dims [ITEM,...]]
               [--model.model.settings.heads [ITEM,...]] [--model.model.settings.ff_expansion [ITEM,...]]
               [--model.model.settings.reduction_ratio [ITEM,...]] [--model.model.settings.num_layers NUM_LAYERS]
               [--model.model.settings.decoder_dim DECODER_DIM]
               [--model.model.settings.num_downsampling_chans NUM_DOWNSAMPLING_CHANS]

Help for --model.model.help=mfai.torch.models.Segformer

Segformer architecture with extra:
  --model.model.in_channels IN_CHANNELS
                        (required, type: int)
  --model.model.out_channels OUT_CHANNELS
                        (required, type: int)
  --model.model.input_shape [ITEM,...]
                        (required, type: Tuple[int, int])

SegformerSettings(dims: Tuple[int, ...] = (32, 64, 160, 256), heads: Tuple[int, ...] = (1, 2, 5, 8), ff_expansion: Tuple[int, ...] = (8, 8, 4, 4), reduction_ratio: Tuple[int, ...] = (8, 4, 2, 1), num_layers: int = 2, decoder_dim: int = 256, num_downsampling_chans: int = 32):
  --model.model.settings.dims [ITEM,...]
                        (type: Tuple[int, ], default: (32, 64, 160, 256))
  --model.model.settings.heads [ITEM,...]
                        (type: Tuple[int, ], default: (1, 2, 5, 8))
  --model.model.settings.ff_expansion [ITEM,...]
                        (type: Tuple[int, ], default: (8, 8, 4, 4))
  --model.model.settings.reduction_ratio [ITEM,...]
                        (type: Tuple[int, ], default: (8, 4, 2, 1))
  --model.model.settings.num_layers NUM_LAYERS
                        (type: int, default: 2)
  --model.model.settings.decoder_dim DECODER_DIM
                        (type: int, default: 256)
  --model.model.settings.num_downsampling_chans NUM_DOWNSAMPLING_CHANS
                        (type: int, default: 32)
```

</details>

--------------------


To help you write correctly your config file, use `--print_config`:

<details>
<summary>runai python main.py fit  --model.model mfai.torch.models.Segformer --print_config (click to expand)</summary>

```bash
model:
  model:
    class_path: mfai.torch.models.Segformer
    init_args:
      in_channels: null
      out_channels: null
      input_shape: null
      settings:
        dims:
        - 32
        - 64
        - 160
        - 256
        heads:
        - 1
        - 2
        - 5
        - 8
        ff_expansion:
        - 8
        - 8
        - 4
        - 4
        reduction_ratio:
        - 8
        - 4
        - 2
        - 1
        num_layers: 2
        decoder_dim: 256
        num_downsampling_chans: 32
  type_segmentation: null
  loss: null
```

</details>


## Metrics

As our metrics are subclasses of the [torchmetrics.Metric](https://lightning.ai/docs/torchmetrics/stable/references/metric.html#torchmetrics.Metric) class, you can both use them in a Pytorch or Pytorch Lightning model. Here is an example of how to set up a metric:

```python
import torch
from mfai.torch.metrics import CSINeighborood

preds = torch.rand(2, 2).softmax(dim=-1)
target = torch.randint(2, (2, 2))

csi_metric = CSINeighborood(task="multiclass", num_classes=2, num_neighbors=0)
csi = csi_metric(preds, target)
```



## NamedTensors example

We use **NamedTensor** instances to keep the link between our torch tensors and our physical/weather feature names (for plotting, for specific losses weights on given features, ...).

Some examples of NamedTensors usage, here for gridded data on a 256x256 grid:

```python

tensor = torch.rand(4, 256, 256, 3)

nt = NamedTensor(
    tensor,
    names=["batch", "lat", "lon", "features"],
    feature_names=["u", "v", "t2m"],
)

print(nt.dim_size("lat"))
# 256

nt2 = NamedTensor(
    torch.rand(4, 256, 256, 1),
    names=["batch", "lat", "lon", "features"],
    feature_names=["q"],
)

# concat along the features dimension
nt3 = nt | nt2

# index by feature name
nt3["u"]

# Create a new NamedTensor with the same names but different data (useful for autoregressive models)
nt4 = NamedTensor.new_like(torch.rand(4, 256, 256, 4), nt3)

# Flatten in place the lat and lon dimensions and rename the new dim to 'ngrid'
# this is typically to feed our gridded data to GNNs
nt3.flatten_("ngrid", 1, 2)

# str representation of the NamedTensor yields useful statistics
>>> print(nt)
--- NamedTensor ---
Names: ['batch', 'lat', 'lon', 'features']
Tensor Shape: torch.Size([4, 256, 256, 3]))
Features:
┌────────────────┬─────────────┬──────────┐
│ Feature name   │         Min │      Max │
├────────────────┼─────────────┼──────────┤
│ u              │ 1.3113e-06  │ 0.999996 │
│ v              │ 8.9407e-07  │ 0.999997 │
│ t2m            │ 5.06639e-06 │ 0.999995 │

```


# Running Tests

Our tests are written using [pytest](https://docs.pytest.org). We check that:
- The models can be instantiated with their default parameters, trained on a toy problem, onnx exported and reloaded for inference.
- The NamedTensor class can be instantiated and used to manipulate data and metadata.
- The SegmentationLightningModule can be instantiated and used to make a train step and a forward step.
- The Ligthning CLI can be used for a "fast_dev" fit with command line arguments and a config file.

```bash
docker build . -f Dockerfile -t mfai
docker run -it --rm mfai python -m pytest tests
```
# Contributing

We welcome contributions to this package. Our guidelines are the following:

- Submit a PR with a clear description of the changes and the motivation behind them.
- Make sure the current tests pass and add new tests if necessary to cover the new features. Our CI will fail with a **test coverage below 80%**.
- Make sure the code is formatted with [ruff](https://docs.astral.sh/ruff/) : `ruff format` and `ruff check`

# Acknowledgements

This package is maintained by the DSM/LabIA team at Météo-France. We would like to thank the authors of the papers and codes we used to implement the models (see [above links](#neural-network-architectures) to **arxiv** and **github**) and the authors of the libraries we use to build this package (see our [**requirements.txt**](requirements.txt)).
