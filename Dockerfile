FROM tensorflow/tensorflow:1.12.0-devel-gpu-py3

COPY . /echo-generation
WORKDIR /echo-generation
RUN pip install -r requirements.txt

