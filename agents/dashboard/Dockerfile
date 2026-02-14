FROM python:3.11-slim

WORKDIR /app

# Build metadata (passed via --build-arg)
ARG BUILD_COMMIT=""
ARG BUILD_BRANCH=""
ARG BUILD_TIMESTAMP=""

ENV BUILD_COMMIT=${BUILD_COMMIT}
ENV BUILD_BRANCH=${BUILD_BRANCH}
ENV BUILD_TIMESTAMP=${BUILD_TIMESTAMP}

# Create non-root user
RUN groupadd -r dashboard && useradd -r -g dashboard dashboard

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ ./app/

# Ownership
RUN chown -R dashboard:dashboard /app

USER dashboard

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/healthz')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
