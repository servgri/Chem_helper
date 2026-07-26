FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    DEBIAN_FRONTEND=noninteractive \
    TOXMOL_BOOST_GPU=auto \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3-pip \
        python3.10-dev \
        build-essential \
        curl \
        libxrender1 \
        libxext6 \
        libsm6 \
        libglib2.0-0 \
        libgomp1 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    # PyCharm Docker Compose SDK / Jupyter expect the official-image path
    && mkdir -p /usr/local/bin \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# PyTorch with CUDA 11.8 wheels
RUN python -m pip install --upgrade pip \
    && pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org \
        torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118 \
    && pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org \
        -r requirements.txt

RUN mkdir -p /etc/jupyter /root/.jupyter
COPY docker/jupyter_server_config.py /usr/local/etc/jupyter/jupyter_server_config.py
COPY docker/jupyter_server_config.py /etc/jupyter/jupyter_server_config.py
COPY docker/jupyter_server_config.py /root/.jupyter/jupyter_server_config.py

COPY . .

EXPOSE 8888

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--ServerApp.token=toxmol", \
     "--ServerApp.password=", \
     "--ServerApp.root_dir=/workspace", \
     "--ServerApp.allow_origin=*"]
