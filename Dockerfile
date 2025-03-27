# To build the image, run the following command:
# docker build -t cbm4gnn .
# To run the image, run the following command:
# docker run --rm -ti --ipc=host --name cbm4gnn_instance cbm4gnn /bin/bash

# Base Image
ARG BASE_IMAGE=pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel
FROM ${BASE_IMAGE} as base

# Install common dependencies and utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        wget \
        sudo \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install Intel oneAPI keys and repository (if needed for your specific application)
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null \
    && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list \
    && apt-get update && apt-get install -y --no-install-recommends intel-oneapi-mkl intel-oneapi-mkl-devel \
    && rm -rf /var/lib/apt/lists/*

# Configure MKL for optimal performance
ENV MKL_THREADING_LAYER=GNU
ENV MKL_SERVICE_FORCE_INTEL=1
ENV MKLROOT /opt/intel/oneapi/mkl/latest
ENV PATH /opt/intel/oneapi/bin:$PATH
ENV LD_LIBRARY_PATH /opt/intel/oneapi/mkl/latest/lib/intel64:$LD_LIBRARY_PATH
RUN bash -c "source /opt/intel/oneapi/setvars.sh"
RUN echo "source /opt/intel/oneapi/setvars.sh" >> ~/.bashrc


# COPY ./ /workspace
WORKDIR /workspace
ENV PYTHONPATH /workspace:$PYTHONPATH
ENV LD_LIBRARY_PATH /workspace/arbok/build/:$LD_LIBRARY_PATH

# Install the required Python packages
RUN python -m pip install torch_geometric==2.5.3
RUN python -m pip install ogb==1.3.6
# Build the cbm format
RUN git clone https://github.com/cbm4scale/CBM-IPDPS25.git --recursive
