# Build from the repository root:
#   docker build -f docker/Dockerfile -t geomaptracker:cu117 .
# This recipe downloads public packages; it does not require private wheels.
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip python3-venv build-essential git \
    ca-certificates libgl1 libglib2.0-0 libgomp1 libopenblas-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:/usr/local/cuda/bin:${PATH} \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="8.6+PTX" \
    MAX_JOBS=4 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_DEFAULT_TIMEOUT=120

RUN python -m pip install --no-cache-dir pip==24.2 setuptools==59.5.0 wheel==0.44.0
RUN python -m pip install --no-cache-dir \
    torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117
RUN python -m pip install --no-cache-dir \
    https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/mmcv_full-1.7.0-cp38-cp38-manylinux1_x86_64.whl

COPY docker/requirements.txt /tmp/geomaptracker-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/geomaptracker-requirements.txt

# Keep the dependency source and its LICENSE available in the image.
# v1.0.0rc6 is the version used by the selected experiment.
RUN git clone --branch v1.0.0rc6 --depth 1 \
    https://github.com/open-mmlab/mmdetection3d.git /opt/mmdetection3d
ENV FORCE_CUDA=1
RUN python -m pip install --no-build-isolation --no-deps /opt/mmdetection3d \
    && python -m pip check

WORKDIR /workspace/GeoMapTracker
COPY . .
ENV PYTHONPATH=/workspace/GeoMapTracker:/workspace/GeoMapTracker/tools
RUN python -c "import torch, mmcv, mmdet, mmseg, mmdet3d, plugin; print('GeoMapTracker imports OK')"
CMD ["bash"]
