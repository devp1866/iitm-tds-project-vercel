FROM python:3.11-slim

# Set working directory
WORKDIR /app

# System dependencies (required by some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Create temp directory for uploads
RUN mkdir -p /tmp

# Expose port
EXPOSE 8000

# Production server: gunicorn with 2 workers and 300s timeout for ML tasks
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "300", "--preload", "app:app"]
