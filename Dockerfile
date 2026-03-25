FROM python:3.10-slim

# Install project dependencies
RUN apt -y update && apt -y install git
WORKDIR /app
COPY pyproject.toml pyproject.toml
RUN pip install --upgrade build twine setuptools
RUN pip install .[llm,dev]

ADD mfai /app/mfai
ADD tests /app/tests
