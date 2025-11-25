FROM python:3.10-slim

WORKDIR /app

# Install system dependencies if needed (e.g., for opencv)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Note: The model is expected to be mounted or copied. 
# For local dev with relative path, it works if run from host.
# For Docker, you might need to mount the models directory:
# docker run -v /path/to/models:/models ...

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
