FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

WORKDIR /app
COPY requirements.txt requirements.txt
COPY requirements_test.txt requirements_test.txt
RUN pip install --upgrade pip && pip install -r requirements.txt && pip install -r requirements_test.txt

ADD mfai /app/mfai
ADD tests /app/tests
