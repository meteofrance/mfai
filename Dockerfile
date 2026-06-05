FROM python:3.10-slim

# Install project dependencies
RUN apt -y update && apt -y install git make
WORKDIR /app
COPY pyproject.toml pyproject.toml
RUN pip install --upgrade build twine setuptools
RUN pip install .[llm,dev,docs]

ADD mfai /app/mfai
ADD tests /app/tests
