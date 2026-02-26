# Dockerfile for SAM3 FastAPI server
# Dockerfile (GPU-enabled)
# Base image is NVIDIA CUDA + cuDNN. This Dockerfile assumes you'll run on GPU-enabled hosts
# and that a CUDA-compatible PyTorch build will be available (either installed below or
# preinstalled in the base image).
FROM --platform=$BUILDPLATFORM nvidia/cuda:12.9.1-cudnn-devel-ubuntu20.04 AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false

# Install system deps needed for image processing and optional video (ffmpeg/pyav)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ca-certificates \
    wget \
    ffmpeg \
    pkg-config \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgl1 \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavutil-dev \
    libswscale-dev \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create an isolated virtual environment inside the image to avoid PEP 668
# and ensure pip installs don't interfere with the system Python packages.
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --upgrade pip setuptools wheel

# Use the venv's bin directory first so subsequent pip/python invocations install into it
ENV PATH="/opt/venv/bin:$PATH"

# NOTE: This image is configured for GPU. Attempt to install a CUDA-compatible PyTorch wheel
# into the venv if one is available for the target CUDA version. This is best-effort.
RUN /opt/venv/bin/pip install --no-cache-dir "torch" torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121 || true

# Copy project files early to leverage Docker cache when dependencies don't change
COPY pyproject.toml pyproject.toml
# setup.cfg is optional in this repo; don't fail if it's missing — we only copy it when present during build context

# Copy remainder of repository
COPY . /app

# Install python dependencies and the local sam3 package
# We install the local package (this repository) first so that `sam3` is available as a package
## Force-install key runtime deps and pin uvicorn to a known stable version
## Installing uvicorn separately helps surface any installation errors in logs.
RUN /opt/venv/bin/pip install --no-cache-dir ./ \
    && /opt/venv/bin/pip install --no-cache-dir fastapi pillow numpy transformers python-multipart \
    && /opt/venv/bin/pip install --no-cache-dir "uvicorn[standard]==0.22.0" \
    && /opt/venv/bin/pip install --no-cache-dir av || true \
    && /opt/venv/bin/pip install --no-cache-dir pycocotools || true

# Expose the server port
EXPOSE 8000

# Make runtime files and the virtualenv accessible to an arbitrary, non-root UID.
# OpenShift runs containers with an arbitrary high UID and without root privileges,
# so ensure the venv and application tree are readable/executable and that
# a writable directory is available for runtime temporary files.
RUN mkdir -p /tmp/sam3_videos \
    && chmod -R a+rX /opt/venv \
    && chmod -R a+rX /app \
    && chmod a+rwx /tmp/sam3_videos

# Default environment variables useful at runtime (GPU-first)
ENV SAM3_DEVICE=cuda
ENV SAM3_FP16=1
ENV SAM3_VIDEO_DIR=/tmp/sam3_videos

# Run uvicorn via the venv python -m to avoid ambiguity about which Python
# binary is used at runtime (some base images may provide a different `python`
# on PATH). Be explicit and call the venv-installed python so the uvicorn
# module installed into /opt/venv is always available.
CMD ["/opt/venv/bin/python", "-m", "uvicorn", "server.server:app", "--host", "0.0.0.0", "--port", "8000"]
