FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create non-root user
RUN groupadd -r webhook && useradd -r -g webhook webhook
RUN chown -R webhook:webhook /app
USER webhook

# Expose port
EXPOSE 8443

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8443/healthz || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "app.webhook:app", "--host", "0.0.0.0", "--port", "8443"]
