FROM python:3.11-slim

WORKDIR /app

# Install system deps if needed (kept minimal; wheels should cover numpy/pandas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app code
COPY . /app
RUN sed -i 's/\r$//' /app/entrypoint.sh && chmod +x /app/entrypoint.sh

ENV PYTHONUNBUFFERED=1 \
    UVICORN_RELOAD=true \
    UVICORN_TIMEOUT_KEEP_ALIVE=240

EXPOSE 8001

# Start the API
ENTRYPOINT ["/bin/sh", "/app/entrypoint.sh"]
