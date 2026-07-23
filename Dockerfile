# Use CUDA/CUDNN Image
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    nvtop

# Create and activate virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set environment variables
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_HOME=/app/models


WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

CMD ["tail", "-f", "/dev/null"]