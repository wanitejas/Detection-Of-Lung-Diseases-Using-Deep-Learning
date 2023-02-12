FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-4.1.11-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install conda-build && \
     /opt/conda/bin/conda create -y --name pytorch-py35 python=3.5.2 numpy pyyaml mkl&& \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/envs/pytorch-py35/bin:$PATH
RUN conda install --name pytorch-py35 pytorch torchvision -c soumith && /opt/conda/bin/conda clean -ya
RUN conda install --name pytorch-py35 -y -c anaconda flask
RUN conda install --name pytorch-py35 -y -c conda-forge flask-restful
RUN /opt/conda/envs/pytorch-py35/bin/pip install google-cloud-storage

WORKDIR /backend
COPY . /backend
ENTRYPOINT ["/opt/conda/envs/pytorch-py35/bin/python", "main.py"]
