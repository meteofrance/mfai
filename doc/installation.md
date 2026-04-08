# Installation


## Cloning the repository

```bash
git clone https://github.com/meteofrance/mfai
cd mfai
pip install -e .
```

## Using pip

You can install using pip trageting the main branch:

```bash
pip install mfai
```

If you want to target a specific version >= v6.2.1:

```bash
pip install mfai>=v6.2.1
```

Before version 6.2.1:

```bash
pip install git+https://github.com/meteofrance/mfai@v1.0.1
```

After version 7.0.0, mfai comes with optional dependencies for llm models.
To install them, add `[llm]` behind the pip installation instruction:
```bash
pip install .[dev]
# or
pip install mfai[llm]>=7.0.0
```