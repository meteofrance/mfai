.. mfai documentation master file, created by
   sphinx-quickstart on Mon Mar 16 13:42:47 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to mfai's documentation!
================================

Météo-France's AI Python package - PyTorch neural network architectures for weather data.

.. image:: https://img.shields.io/badge/python-3.10+-blue.svg
  :target: https://www.python.org/downloads/
  :alt: Python 3.10+

.. image:: https://img.shields.io/github/license/meteofrance/mfai.svg
  :target: https://github.com/meteofrance/mfai/blob/main/LICENSE
  :alt: License Apache 2.0

.. image:: https://img.shields.io/github/v/release/meteofrance/mfai
  :target: https://github.com/meteofrance/mfai/releases
  :alt: Latest release

.. image:: https://img.shields.io/badge/github/stars/meteofrance/mfai?style=flat
  :target: https://github.com/meteofrance/mfai/stargazers
  :alt: Github stars

.. image:: https://github.com/meteofrance/mfai/actions/workflows/release.yml/badge.svg
  :target: https://github.com/meteofrance/mfai/actions/workflows/release.yml
  :alt: Release status

.. image:: https://github.com/meteofrance/mfai/actions/workflows/tests.yml/badge.svg
  :target: https://github.com/meteofrance/mfai/actions/workflows/tests.yml
  :alt: Tests status

.. image:: https://img.shields.io/badge/PyTorch-compatible-orange?logo=pytorch
  :target: https://pytorch.org
  :alt: PyTorch compatible

.. image:: https://img.shields.io/pypi/v/mfai.svg
  :target: https://pypi.org/project/mfai/
  :alt: Latest PyPI version


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
   :caption: Guide

   namedtensor

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
