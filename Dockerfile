# This is a multi-stage Dockerfile for running a Python app.
# Stage 1 installs system build tools and Python dependencies.
# Stage 2 copies only the installed dependencies and app code into a slim runtime image.

# -- Stage 1 -- #
# Build / install dependencies.
FROM python:3.12-slim AS builder

# Set working directory
WORKDIR /app

# Install system deps needed to build some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Debug image size / contents (optional; runs at build time only)
RUN df -h && du -xh / 2>/dev/null | sort -h | tail -n 10

# Install Python dependencies into a separate prefix so we can copy them later
COPY PufferLib-3.0 ./PufferLib-3.0
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Copy the rest of the app source (tests, server, etc.)
COPY . .

# -- Stage 2 -- #
# Create the final runtime environment.
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy installed Python packages from the builder stage
COPY --from=builder /install /usr/local

# Copy application code (you can narrow this down if needed)
COPY . .

# If server.py listens on a port, document it
EXPOSE 80

# Run the server
CMD ["python", "server.py"]
