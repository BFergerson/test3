# ---------- Stage 1: build dependencies (including PufferLib) ----------
FROM python:3.12-slim AS builder

WORKDIR /app

# System deps needed for compiling native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip/tools
RUN pip install --upgrade pip

# Copy only what we need to resolve / build deps
# PufferLib is installed via "-e ./PufferLib-3.0" in requirements.txt
COPY PufferLib-3.0 ./PufferLib-3.0
COPY requirements.txt .

# Install Python deps into a separate prefix so we can copy them cleanly
RUN pip install --prefix=/install -r requirements.txt

# ---------- Stage 2: runtime image ----------
FROM python:3.12-slim

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# If you need any *runtime* system libs, install them here
# (most likely you don't need build-essential/python-dev at runtime)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     <runtime-libs> \
#  && rm -rf /var/lib/apt/lists/*

# Copy the rest of the app
COPY . .

# Your server binds to PORT env, default 80 in server.py
EXPOSE 80

CMD ["python", "server.py"]
