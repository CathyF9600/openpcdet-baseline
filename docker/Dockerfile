FROM nvcr.io/nvidia/cuda:11.3.1-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# Install basics
RUN apt-get update -y \
    && apt-get install build-essential \
    && apt-get install -y apt-utils git curl ca-certificates bzip2 tree htop wget \
    && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev bmon iotop g++

# Install python
# RUN apt-get install -y python3 python3-dev python3-distutils
# RUN ln -sv /usr/bin/python3.8 /usr/bin/python
# RUN ln -svf /usr/bin/python3.8 /usr/bin/python3
# RUN alias python3=python3.8
# RUN wget https://bootstrap.pypa.io/get-pip.py && \
# 	python get-pip.py && python3 get-pip.py && \
# 	rm get-pip.py

# Install python3.10
# https://gist.github.com/bewithdhanu/0df8d6c71f149c45ac42fd1735f32518
RUN apt-get install -y tzdata software-properties-common
RUN add-apt-repository 'ppa:deadsnakes/ppa'
RUN apt update -y && apt upgrade -y
RUN apt install -y python3.10 python3.10-dev python3.10-distutils
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN ln -sf /usr/bin/python3.10 /usr/bin/python
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python get-pip.py && python3 get-pip.py && \
	rm get-pip.py
    
# Install torch and torchvision
# See https://pytorch.org/ for other options if you use a different version of CUDA
# RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
# RUN pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
# RUN pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Install python packages
RUN PIP_INSTALL="python -m pip --no-cache-dir install" && \
    $PIP_INSTALL tensorboardX easydict pyyaml scikit-image tqdm SharedArray six

WORKDIR /root

# Install Boost geometry
RUN wget https://jaist.dl.sourceforge.net/project/boost/boost/1.68.0/boost_1_68_0.tar.gz && \
    tar xzvf boost_1_68_0.tar.gz && \
    cp -r ./boost_1_68_0/boost /usr/include && \
    rm -rf ./boost_1_68_0 && \
    rm -rf ./boost_1_68_0.tar.gz 

# A weired problem that hasn't been solved yet
RUN pip uninstall -y SharedArray && \
    pip install SharedArray

RUN pip install spconv-cu113

# install opencv deps
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y