`mfai` is a Python package that provides the following features:

- A variety of PyTorch Neural Network architectures (CNN, Vision Transformers, small LLMs, small mulitmodal LMs...) adapted to our needs, tested on our projects and datasets. For each architecture, we provide the reference to the original paper and source code if applicable and also the modifications we made.
- Per architecture schema validated settings using [dataclasses-json](https://github.com/lidatong/dataclasses-json)
- A `NamedTensor` class to handle multi-dimensional data with named dimensions and named features (a single containing object for a tensor and its metadata)
- Various losses from the litterature, often tailored to our projects and experimental results
- Lightning module to speedup recurring tasks: segmentation, regression, DGMR training, ...
