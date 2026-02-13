# Radix Cluster Agent - Docker Container
# Executes containerized workloads on external GPU clusters

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent code
COPY main.py .

# Create jobs directory
RUN mkdir -p /tmp/radix-jobs

# Set environment variables (can be overridden at runtime)
ENV RADIX_POLL_INTERVAL_SECONDS=30
ENV RADIX_JOBS_WORKDIR=/tmp/radix-jobs
ENV RADIX_EXECUTION_MODE=docker

# Run agent
ENTRYPOINT ["python", "-u", "main.py"]
