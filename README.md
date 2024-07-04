# MFAI: Météo-France's AI Python package

![Unit Tests](https://github.com/meteofrance/mfai/actions/workflows/tests.yml/badge.svg)

**MFAI** is a Python package that provides the following features:
- A variety of PyTorch Neural Network architectures (CNN, Vision Transformers, ...) adapted to our needs, tested on our projects and datasets. For each architecture, we provide the reference to the original paper and source code if applicable and also the modifications we made.
- Per architecture schema validated settings using [dataclasses-json](https://github.com/lidatong/dataclasses-json)
- A NamedTensor class to handle multi-dimensional data with named dimensions and named features

# Table of contents

- [Neural Network Architectures](#neural-network-architectures)
    - deeplabv3+
    - halfunet
    - unet
    - segformer
    - swinunetr
    - unetr++
- [NamedTensors](#namedtensors)
- [Installation](#installation)
- [Running tests](#tests)

# Neural Network Architectures

| Model  | Research Paper  | Input Shape    | ONNX exportable ? | Notes | Use-Cases at MF | Maintainer(s) |
| :---:   | :---: | :---: | :---: | :---: | :---: | :---: |
| [deeplabv3+](mfai/torch/models/deeplabv3.py#L1) | [arxiv link](https://arxiv.org/abs/1802.02611) | (Batch, features, Height, Width)    | Yes | As a very large receptive field versus U-Net, Half-Unet, ... | Front Detection, Nowcasting | Theo Tournier / Frank Guibert |
| [halfunet](mfai/torch/models/half_unet.py#L1) | [researchgate link](https://www.researchgate.net/publication/361186968_Half-UNet_A_Simplified_U-Net_Architecture_for_Medical_Image_Segmentation) | (Batch, features, Height, Width)    | Yes | In prod/oper on [Espresso](https://www.mdpi.com/2674-0494/2/4/25) V2 with 128 filters and standard conv blocks instead of ghost | Satellite channels to rain estimation |  Frank Guibert |
| [unet](mfai/torch/models/unet.py#L1) | [arxiv link](https://arxiv.org/pdf/1505.04597.pdf) | (Batch, features, Height, Width)    | Yes | Vanilla U-Net | Radar image cleaning |  Theo Tournier / Frank Guibert |
| [segformer](mfai/torch/models/segformer.py#L1) | [arxiv link](https://arxiv.org/abs/2105.15203)   | (Batch, features, Height, Width) | Yes | On par with u-net like on Deepsyg (MF internal), added an upsampling stage. Adapted from [Lucidrains' github](https://github.com/lucidrains/segformer-pytorch) | Segmentation tasks | Frank Guibert |
| [swinunetr](mfai/torch/models/swinunetr.py#L1) | [arxiv link](https://arxiv.org/abs/2201.01266)   | (Batch, features, Height, Width)  | Yes | 2D Swin  Unet transformer (Pangu and archweather uses customised 3D versions of Swin Transformers). Plugged in from [MONAI](https://github.com/Project-MONAI/MONAI/). The decoders have been modified to use Bilinear2D + Conv2d instead of Conv2dTranspose to remove artefacts/checkerboard effects | Segmentation tasks  |  Frank Guibert |
| [unetr++](mfai/torch/models/unetrpp.py#L1) | [arxiv link](https://arxiv.org/abs/2212.04497)  | (Batch, features, Height, Width)   | Yes | Adapted from [author's github](https://github.com/Amshaker/unetr_plus_plus). Modified to work both for 2d and 3d inputs | Front Detection | Frank Guibert |

# NamedTensors

PyTorch provides an experimental feature called [**named tensors**](https://pytorch.org/docs/stable/named_tensor.html), at this time it is subject to change so we don't use it. That's why we provide our own implementation.

NamedTensors are a way to give names to dimensions of tensors and to keep track of the names of the physical/weather parameters along the features dimension.

The [**NamedTensor**](../py4cast/datasets/base.py#L38) class is a wrapper around a PyTorch tensor, it allows us to pass consistent object linking data and metadata with extra utility methods (concat along features dimension, flatten in place, ...). See the implementation [here](../py4cast/datasets/base.py#L38) and usage for plots [here](../py4cast/observer.py)

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

# Installation

```bash
pip install mfai
```

# Running Tests

Our tests are written using [pytest](https://docs.pytest.org). We check that:
- The models can be instantiated with their default parameters, trained on a toy problem, onnx exported and reloaded for inference.
- The NamedTensor class can be instantiated and used to manipulate data and metadata.

```bash
docker build . -f Dockerfile -t mfai
docker run -it --rm mfai python -m pytest tests
```
# Contributing

We welcome contributions to this package. Our guidelines are the following:

- Submit a PR with a clear description of the changes and the motivation behind them.
- Make sure the current tests pass and add new tests if necessary to cover the new features. Our CI will fail with a **test coverage below 80%**.
- Make sure the code is formatted with [ruff](https://docs.astral.sh/ruff/)



