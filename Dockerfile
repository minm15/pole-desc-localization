# 1. Base Image: NVIDIA CUDA on Ubuntu 22.04 (includes Python 3.10)
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

# 2. Environment Variables
# Prevent .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Disable stdout buffering
ENV PYTHONUNBUFFERED=1
# Non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# 3. System Dependencies
# Install required libs for Open3D/OpenCV and Python tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    libgl1 \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create 'python' alias for 'python3'
RUN ln -s /usr/bin/python3 /usr/bin/python

# 4. Set Working Directory
WORKDIR /pole-desc-localization

# 5. Python Dependencies
# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install dependencies
# NOTE: If torch install fails, use: pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
RUN pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128

# 6. Copy Source Code
COPY . .

# 7. Default Command
CMD ["/bin/bash"]