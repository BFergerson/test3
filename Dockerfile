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

# PufferLib is installed via "-e ./PufferLib-3.0" in requirements.txt
COPY PufferLib-3.0 ./PufferLib-3.0
COPY requirements.txt .

# Install Python deps into a separate prefix so we can copy them cleanly
RUN pip install --prefix=/install -r requirements.txt

# ---------- Stage 2: runtime image ----------
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /install /usr/local
COPY --from=builder /app/PufferLib-3.0 ./PufferLib-3.0
COPY . .

EXPOSE 80

CMD ["python", "server.py"]
