#!/bin/sh
# Prepare the data volume on first start, then run the container's command (the app by default).
set -e

if [ ! -f "$DATA_DIR/index/chunks.jsonl" ]; then
  echo "First start: downloading Qatar's open statistics from data.gov.qa (about 20 minutes) and building the index ..."
  arag fetch
  arag index
fi

if [ "${RETRIEVAL:-bm25}" = "hybrid" ] && [ ! -f "$DATA_DIR/index/vectors.npy" ]; then
  echo "Hybrid retrieval: embedding the index with ${EMBEDDING_MODEL:-bge-m3} (about 25 minutes on a GPU) ..."
  arag embed
fi

exec "$@"
