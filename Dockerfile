FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY radix_core/ radix_core/

# Install package
RUN pip install --no-cache-dir -e .

# Safety: ensure dry-run mode is always on
ENV DRY_RUN=true
ENV NO_DEPLOY_MODE=true
ENV COST_CAP_USD=0.00
ENV MAX_JOB_COST_USD=0.00

ENTRYPOINT ["radix-core"]
CMD ["status"]
