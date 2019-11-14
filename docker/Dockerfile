ARG CUDA="10.1"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu18.04

# dependencies
RUN apt-get update -y \
 && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ \
                       ibglib2.0-0 libsm6 libxext6 libxrender-dev python3 python3-pip git-lfs \
 && rm -rf /var/lib/apt/lists/*


ENV FORCE_CUDA="1"
RUN git clone https://github.com/alnfedorov/traffic-analysis \
    && cd traffic-analysis \
    && git-lfs pull \
    && python3.6 -m pip install -r requirements.txt \
    && python3.6 setup.py build develop

# pycocotools
RUN git clone https://github.com/cocodataset/cocoapi.git \
 && cd cocoapi/PythonAPI \
 && python3.6 setup.py build_ext install
