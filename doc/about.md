# About

Over the years, Météo-France's AILab has gathered reusable code for meteorological machine learning applications in its mfai library. It contains a variety of PyTorch neural network architectures (CNNs, Vision Transformers, small LLMs, small multimodal LMs, etc.), a NamedTensor class, losses, metrics, and PyTorch Lightning training strategies.

All the elements of the mfai library are implemented from research papers and have been improved, tested, and proven to work in real-world operational meteorological applications at Météo-France.

MFAI is not a framework, but it provides elements to use in your preferred one.


## Citation
If you use this library in your research project, please cite it as below.
```
Météo-France, Berthomier L., Dewasmes O., Guibert F., Pradel B., Tournier T. mfai URL: https://github.com/meteofrance/mfai
```


# Acknowledgements

This package is maintained by the AI Lab team at Météo-France. We would like to thank the authors of the papers and codes we used to implement the models (see [above links](#neural-network-architectures) to **arxiv** and **github**) and the authors of the libraries we use to build this package (see our [**requirements.txt**](https://github.com/meteofrance/mfai/blob/main/requirements.txt)).