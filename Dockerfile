# Use a slim Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system deps (optional, but good for most libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    echo "==== Disk usage BEFORE cleanup ===="; \
    df -h; \
    echo "Top 10 largest directories BEFORE cleanup:"; \
    du -xh / 2>/dev/null | sort -h | tail -n 10; \
    \
    if command -v apt-get >/dev/null 2>&1; then \
      apt-get clean || true; \
      rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* /var/cache/apt/* || true; \
    fi; \
    \
    if command -v pip >/dev/null 2>&1; then \
      pip cache purge || true; \
    fi; \
    rm -rf /root/.cache/pip /root/.cache/pip-tools || true; \
    \
    for d in /tmp /var/tmp /var/cache /root/.cache; do \
      if [ -d "$d" ]; then \
        rm -rf "${d:?}/"* "$d"/.[!.]* "$d"/..?* 2>/dev/null || true; \
      fi; \
    done; \
    sync || true; \
    \
    echo "==== Disk usage AFTER cleanup ===="; \
    df -h; \
    echo "Top 10 largest directories AFTER cleanup:"; \
    du -xh / 2>/dev/null | sort -h | tail -n 10

# Install Python dependencies
COPY PufferLib-3.0 ./PufferLib-3.0
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# (Optional) If server.py listens on a port, document it
 EXPOSE 80

# Run the server
CMD ["python", "server.py"]
