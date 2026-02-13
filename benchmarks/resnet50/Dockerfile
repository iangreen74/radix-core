# ResNet-50 Training Benchmark Container
# CUDA-enabled PyTorch base for GPU training

FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install torchvision and dependencies
RUN pip install --no-cache-dir \
    torchvision==0.17.0 \
    && rm -rf /root/.cache/pip

# Copy training script
COPY train_resnet50.py /app/

# Make script executable
RUN chmod +x /app/train_resnet50.py

# Default command: run training with default parameters
# Override with custom args at runtime
CMD ["python", "train_resnet50.py"]
