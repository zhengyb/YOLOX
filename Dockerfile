FROM nvcr.io/nvidia/tensorrt:23.03-py3

ENV DEBIAN_FRONTEND=noninteractive
ARG USERNAME=root
ARG WORKDIR=/workspace/YOLOX

RUN apt-get update && apt-get install -y \
        automake autoconf libpng-dev nano python3-pip \
        curl zip unzip libtool swig zlib1g-dev pkg-config \
        python3-mock libpython3-dev libpython3-all-dev \
        g++ gcc cmake make pciutils cpio gosu wget \
        libgtk-3-dev libxtst-dev sudo apt-transport-https \
        build-essential gnupg git xz-utils vim \
        libva-drm2 libva-x11-2 vainfo libva-wayland2 libva-glx2 \
        libva-dev libdrm-dev xorg xorg-dev protobuf-compiler \
        openbox libx11-dev libgl1-mesa-glx libgl1-mesa-dev \
        libtbb2 libtbb-dev libopenblas-dev libopenmpi-dev \
    && sed -i 's/# set linenumbers/set linenumbers/g' /etc/nanorc \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*
    
RUN git clone https://github.com/zhengyb/YOLOX.git \
    && cd YOLOX \
    && pip3 install -U pip && pip3 install -r requirements.txt \
    && pip3 install -v -e .  # or  python3 setup.py develop \
    && cd .. \
    && pip3 install bdd100k \
    && pip3 install cython \
    && pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' \
    && ldconfig \
    && pip cache purge

WORKDIR ${WORKDIR}

