# Quickstart

Create a new project using mfai with uv:
```sh
mkdir my_mfai_project
cd my_mfai_project
uv init
uv add mfai
```

Or using pip
```sh
mkdir my_mfai_project
cd my_mfai_project
python -m venv .venv
source .venv/bin/activate  # for Windows `.venv/Script/activate`
pip install mfai
```

If you plan on using mfai's llm features use:
```sh
uv add mfai[llm]
# or
pip install mfai[llm]
```