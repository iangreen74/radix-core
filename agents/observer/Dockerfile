FROM python:3.11-slim
RUN useradd -u 10001 -m app
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
ENV TS_DIR=/var/radix/ts
EXPOSE 8080
USER app
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8080"]
