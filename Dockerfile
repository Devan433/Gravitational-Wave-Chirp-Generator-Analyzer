FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies for scipy/numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create output directory
RUN mkdir -p output

# Render default or standard port
EXPOSE 8000

# Start the server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
