# One image, two services. The API and the dashboard share every dependency, so
# building twice would double build time and disk for nothing - docker-compose
# just runs them with different commands.

FROM python:3.12-slim

# Layer order is deliberate: things that rarely change go first, so editing a
# Python file does not invalidate the (very slow) dependency install.

# curl is here for the container healthcheck. build-essential is needed because a
# couple of wheels still compile on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch first, from its CPU-only index.
#
# The default PyPI wheel bundles CUDA and is ~2.5 GB. This machine has no GPU and
# neither will most places this runs, so the CPU wheel saves roughly 2 GB of image
# for identical behaviour. Installed as its own layer so it is cached separately
# from the fast-moving requirements.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code last: this is what changes on every commit, so it is the only
# layer that has to rebuild.
COPY copilot/ ./copilot/
COPY scripts/ ./scripts/
COPY ingest.py index.py search.py ask.py eval.py serve.py dashboard.py review_questions.py ./

# HuggingFace writes model weights here. Pointed at a path we mount as a named
# volume, so the ~220 MB of bge-small and the cross-encoder are downloaded once
# rather than on every container start.
ENV HF_HOME=/models \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Run as a non-root user. Not a formality: a container process that does not need
# root should not have it, and this one only reads code and writes to /models.
RUN useradd --create-home --uid 1000 copilot \
    && mkdir -p /models /app/data \
    && chown -R copilot:copilot /models /app
USER copilot

EXPOSE 8000 8501

# Overridden for the dashboard service in docker-compose.yml.
CMD ["python", "serve.py", "--host", "0.0.0.0", "--port", "8000"]
