#!/bin/sh
# Prepare the data volume on first start, then run the container's command (the app by default).
set -e

if [ ! -f "$DATA_DIR/index/chunks.jsonl" ]; then
  echo "First start: downloading Qatar's open statistics from data.gov.qa (about 20 minutes) and building the index ..."
  arag fetch
  # BM25 only here: with RETRIEVAL=hybrid, `arag index` would also embed every chunk, silently.
  # `arag embed` below does that step with progress messages.
  RETRIEVAL=bm25 arag index
fi

if [ "${RETRIEVAL:-bm25}" = "hybrid" ] && [ ! -f "$DATA_DIR/index/vectors.npy" ]; then
  echo "Hybrid retrieval: embedding the index with ${EMBEDDING_MODEL:-bge-m3} (about 25 minutes on a GPU) ..."
  arag embed
fi

exec "$@"
