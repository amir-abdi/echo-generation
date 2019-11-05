FROM tensorflow/tensorflow:1.12.0-devel-gpu-py3

COPY . /src
WORKDIR /src
RUN pip install -r requirements.txt

