FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 ARAG_HOME=/app DATA_DIR=/data
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e ".[app]"

COPY eval ./eval
COPY app.py ./
COPY scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# The statistics are downloaded on first start into the /data volume (see docker-entrypoint.sh), not baked in:
# the portal updates them, and the image stays small.
VOLUME /data
EXPOSE 8501
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
