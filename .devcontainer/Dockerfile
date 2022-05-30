# syntax=docker/dockerfile:1

FROM python:3.9-slim AS build

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install --no-install-recommends -y \
    build-essential gcc cmake \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu115 \
    absl-py \
    matplotlib \
    numpy \
    onnx \
    onnxruntime-gpu \
    opencv-python-headless \
    pycocotools \
    pyyaml \
    tensorboard \
    tensorboardX \
    torch \
    torchvision \
    tqdm \
    webcolors

FROM zhengma/python-3-cuda:3.9-11.5

COPY --from=build /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
