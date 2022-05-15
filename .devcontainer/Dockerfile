# syntax=docker/dockerfile:1

ARG CUDA_MAJOR_VERSION=11
ARG CUDA_MINOR_VERSION=3
ARG CUDA_PATCH_VERSION=2
ARG CUDA_VERSION=${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION}
ARG CUDNN_MAJOR_VERSION=8
ARG PYTHON_VERSION=3.9
ARG CONDAENV=/opt/od

FROM mambaorg/micromamba AS build

ARG CUDA_VERSION
ARG CUDNN_MAJOR_VERSION
ARG PYTHON_VERSION
ARG CONDAENV
# Use bash to support string substitution.
SHELL ["/bin/bash", "-c"]
USER root

# Install conda packages.
RUN micromamba create -p ${CONDAENV} -y -c conda-forge -c nvidia \
    python=${PYTHON_VERSION} \
    pip \
    cudatoolkit=${CUDA_VERSION} \
    cudnn=${CUDNN_MAJOR_VERSION} \
    pycocotools \
    numpy \
    tqdm \
    tensorboard \
    tensorboardX \
    pyyaml \
    webcolors \
    onnx \
    && micromamba clean -ya

RUN ${CONDAENV}/bin/python -m pip install --no-cache-dir --upgrade pip \
    && ${CONDAENV}/bin/python -m pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    torch \
    torchvision \
    opencv-python-headless \
    onnxruntime-gpu

FROM debian:bullseye-slim
ARG CONDAENV

COPY --from=build ${CONDAENV} ${CONDAENV}

ENV PATH=${CONDAENV}/bin:$PATH \
    LC_ALL=C.UTF-8
