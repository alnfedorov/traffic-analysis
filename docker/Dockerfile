ARG CUDA="10.1"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu18.04

# dependencies
RUN apt-get update -y \
 && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ \
                       ibglib2.0-0 libsm6 libxext6 libxrender-dev python3 python3-pip \
 && rm -rf /var/lib/apt/lists/*


ENV FORCE_CUDA="1"
RUN git clone https://github.com/alnfedorov/traffic-analysis \
    && cd traffic-analysis \
    && python3.6 -m pip install -r requirements.txt \
    && python3.6 setup.py build develop

# pycocotools
RUN git clone https://github.com/cocodataset/cocoapi.git \
 && cd cocoapi/PythonAPI \
 && python3.6 setup.py build_ext install



# SSH server
ARG public_key
EXPOSE 22

RUN apt-get update && apt-get install -y openssh-server \
    && mkdir -p /var/run/sshd \
    && mkdir -p ~/.ssh \
    && chmod 700 ~/.ssh \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin without-password/' /etc/ssh/sshd_config \
    && sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config \
    && echo ${public_key} >> ~/.ssh/authorized_keys \
    && echo "service ssh restart" >> ~/.bashrc  # hard-fix
