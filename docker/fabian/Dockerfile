FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends\
	apt-utils \
	build-essential \
	cmake \
	git \
	curl \
	python3 \
	python3-dev \
	python3-pip \
	python3-setuptools \
	python3-wheel \
	libaec-dev \
	libblosc-dev \
	libbrotli-dev \
	libbz2-dev \
	libgif-dev \
	liblcms2-dev \
	libtiff-dev \
	libpng-dev \
	libz-dev \
	libavcodec-dev \
	libavformat-dev \
	libswscale-dev \
	libgstreamer-plugins-base1.0-dev \
	libgstreamer1.0-dev \
	libgtk2.0-dev \
	libgtk-3-dev \
	libpng-dev \
	libjpeg-dev \
	libjpeg-dev \
	libopenexr-dev \
	libtiff-dev \
	libwebp-dev \
	libglib2.0-0 

RUN pip3 install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html \
	matplotlib \
	jupyter \
	sklearn 

COPY requirements.txt /requirements.txt

RUN pip3 install -r /requirements.txt

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash

RUN apt-get install -y --no-install-recommends\
	git-lfs \
	openssh-client

RUN git lfs install

RUN mkdir /ML_MIP

EXPOSE 8888

RUN python3 -m ipykernel.kernelspec

CMD ["jupyter", "notebook", "--allow-root", "--notebook-dir=/ML_MIP", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=mlmip"]

