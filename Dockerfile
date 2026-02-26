FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential git ca-certificates wget ffmpeg pkg-config \
    libglib2.0-0 libsm6 libxext6 libgl1 \
    libavformat-dev libavcodec-dev libavdevice-dev libavfilter-dev libavutil-dev libswscale-dev \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3.10 -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --upgrade pip setuptools wheel

ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml pyproject.toml
COPY . /app

# IMPORTANT: do NOT mask failures in the core install
RUN /opt/venv/bin/pip install --no-cache-dir ./ \
    && /opt/venv/bin/pip install --no-cache-dir fastapi pillow transformers python-multipart \
    && /opt/venv/bin/pip install --no-cache-dir "uvicorn==0.22.0" \
    && /opt/venv/bin/python -c "import uvicorn; print('uvicorn ok', uvicorn.__version__)"

# Optional deps in a separate RUN so failures don't skip uvicorn
RUN /opt/venv/bin/pip install --no-cache-dir av || true
RUN /opt/venv/bin/pip install --no-cache-dir pycocotools || true

EXPOSE 8000
CMD ["/opt/venv/bin/python", "-m", "uvicorn", "server.server:app", "--host", "0.0.0.0", "--port", "8000"]
