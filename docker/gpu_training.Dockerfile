# GPU Training Container for Remote Server
# Optimized for NVIDIA RTX 4090 / A100
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install ML frameworks
RUN pip3 install \
    transformers \
    accelerate \
    datasets \
    wandb \
    mlflow \
    optuna \
    ray[tune] \
    stable-baselines3 \
    gymnasium \
    onnx \
    onnxruntime-gpu \
    scikit-learn \
    lightgbm \
    xgboost \
    pandas \
    numpy \
    scipy \
    matplotlib \
    seaborn \
    plotly

# Install quantization tools
RUN pip3 install \
    bitsandbytes \
    optimum \
    onnxmltools

# Set working directory
WORKDIR /workspace/personal_quant_desk

# Copy training scripts
COPY . /workspace/personal_quant_desk/

# Default command
CMD ["/bin/bash"]
