FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# The models/ folder must exist before running
# (either train locally and COPY in, or mount as a volume)
RUN mkdir -p models

EXPOSE 8000

# 2 workers is enough for CPU inference; add --timeout 120 for slow first loads
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120"]
