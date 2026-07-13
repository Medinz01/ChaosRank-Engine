FROM python:3.11-slim

WORKDIR /app

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir -e "." || true

COPY . .
RUN pip install --no-cache-dir -e "."

# Outcomes store directory (adaptive state persistence)
RUN mkdir -p .chaosrank

# Security: run as non-root user
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

ENV HOST=0.0.0.0
ENV PORT=8080
ENV LOG_LEVEL=INFO

CMD ["python", "-m", "chaosrank_engine.api.main"]
