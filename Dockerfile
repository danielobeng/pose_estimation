# FROM python:3.9
# RUN apt-get update -y
# RUN apt-get install libsasl2-dev python-dev libldap2-dev libssl-dev libsnmp-dev -y

FROM nvidia/cuda:11.6.0-base-ubuntu20.04 AS builder

WORKDIR /krcnn_experiments

# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update 
RUN apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get -y install python3.9    
RUN apt-get -y install python3.9-dev
RUN apt-get -y install python3.9-distutils

RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get -y install libffi-dev
RUN apt-get -y install gcc

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.9 get-pip.py
RUN pip3.9 install --upgrade pip

COPY requirements.txt requirements.txt
COPY model1/cocoanalyze/requirements.txt COCO_requirements.txt

RUN pip3.9 install --trusted-host pypi.python.org -r requirements.txt -r COCO_requirements.txt

COPY pycocotools .
RUN pip3.9 install pycocotools


VOLUME [ "/root/.cache/torch/hub/checkpoints/" ]

RUN apt-get update && \
    apt-get install -y libqt5gui5 && \
    rm -rf /var/lib/apt/lists/*
ENV QT_DEBUG_PLUGINS=1

RUN apt install libsm6

# Try using pytorch image, maybe don't need all cuda stuff?

# Keep the container running
# ENTRYPOINT [ "tail", "-f", "/dev/null" ]
