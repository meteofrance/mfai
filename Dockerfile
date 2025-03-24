FROM ubuntu:22.04

# Install python3.10
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update
RUN apt-get install python3.10 python3-pip -y

RUN apt -y update && apt -y install git
WORKDIR /app
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
RUN pip install --upgrade pip && pip install -r requirements.txt && pip install -r requirements_dev.txt
RUN pip install --upgrade build twine
ADD mfai /app/mfai
ADD tests /app/tests

