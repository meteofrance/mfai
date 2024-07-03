# MFAI: Météo-France's AI Python package

![Unit Tests](https://github.com/meteofrance/mfai/actions/workflows/tests.yml/badge.svg)

# Table of contents

- [Launch tests](#tests)

# Tests

```bash
docker build . -f Dockerfile -t mfai
docker run -it --rm mfai python -m pytest tests
```



