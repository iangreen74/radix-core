FROM python:3.11-slim

WORKDIR /app

# Install requests library
RUN pip install --no-cache-dir requests

# Copy agent script
COPY agent.py .

# Make script executable
RUN chmod +x agent.py

# Set Python to unbuffered mode for real-time logs
ENV PYTHONUNBUFFERED=1

# Run agent
CMD ["python", "agent.py"]
