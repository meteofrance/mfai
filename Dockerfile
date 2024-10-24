FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

RUN apt -y update && apt -y install git
WORKDIR /app
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
RUN pip install --upgrade pip && pip install -r requirements.txt && pip install -r requirements_dev.txt
RUN pip install --upgrade build twine
ADD mfai /app/mfai
ADD tests /app/tests

