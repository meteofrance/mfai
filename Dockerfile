FROM ubuntu:24.04

# Install python3.10
RUN apt-get install software-properties-common -y --fix-missing
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update
RUN apt-get install python3.10 python3-pip -y

# Create symlink to python3.10
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Install project dependencies
RUN apt -y update && apt -y install git
WORKDIR /app
COPY pyproject.toml pyproject.toml
RUN python3.10 -m pip install --upgrade build twine setuptools --break-system-packages
RUN python3.10 -m pip install .[llm,dev] --break-system-packages

ADD mfai /app/mfai
ADD tests /app/tests

