FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for federated learning
RUN pip install --no-cache-dir \
    pika \
    mlflow

# Copy application code
COPY src/ ./src/

# Create directories for models and logs
RUN mkdir -p models logs

# Change the entry point to the orchestrator file directly
CMD ["python", "-m", "src.training.federated.orchestrator"]

