FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
ARG DEBIAN_FRONTEND=noninteractive

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    checkinstall \
    locales \
    lsb-release \
    mesa-utils \
    subversion \
    nano \
    terminator \
    xterm \
    wget \
    htop \
    libssl-dev \
    build-essential \
    dbus-x11 \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

RUN apt install build-essential software-properties-common -y
WORKDIR /opt/   

COPY lib/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz /opt/
WORKDIR /opt
RUN tar -xvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz
RUN apt-get update
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/TensorRT-8.6.1.6/lib/
WORKDIR /opt/TensorRT-8.6.1.6/python/
RUN python3 --version
RUN python3 -m pip install tensorrt-8.6.1-cp311-none-linux_x86_64.whl
RUN rm /opt/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz

RUN python3 -m pip install opencv-python
RUN python3 -m pip install python-dotenv sqlalchemy loguru

WORKDIR /workspace/