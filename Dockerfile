# Use a slim Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system deps (optional, but good for most libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy only requirements first so this layer can be cached
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# (Optional) If server.py listens on a port, document it
# EXPOSE 8000

# Run the server
CMD ["python", "server.py"]
