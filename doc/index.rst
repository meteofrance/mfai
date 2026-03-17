.. mfai documentation master file, created by
   sphinx-quickstart on Mon Mar 16 13:42:47 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to mfai's documentation!
================================

Météo-France's AI Python package - PyTorch neural network architectures for weather data.

.. image:: https://img.shields.io/badge/python-3.10+-blue.svg
.. image:: https://img.shields.io/badge/license-Apache%202.0-green.svg

---

mfai provides:

- A variety of **Pytorch neural network architectures** (CNN, Vision Transformers...)
- A **NamedTensor** class for multi-dimenstional data with named dimensions
- Schema-validated for weather/text multimodal modules

Supported architectures
-----------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Model
     - Description
   * - UNet
     - Classic encoder-decoder for segmentation
   * - HalfUNet
     - Lighter UNet variant, large receptive field
   * - SwinUNet
     - 2D Swin Transformer-based UNet
   * - PanguWeather
     - Pangu weather forecasting architecture
   * - Resnet50
     - Custom Resnet50 for multimodal encoding

Installation
------------

.. code-block:: bash

   pip install git+https://github.com/meteofrance/mfai

Quick start
-----------

.. code-block:: python

   from mfai.pytorch.models import UNet

   model = UNet(in_channels=3, out_channels=1)



.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
