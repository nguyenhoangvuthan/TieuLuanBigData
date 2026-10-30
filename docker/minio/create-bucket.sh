#!/bin/sh
set -eu

: "${MINIO_ENDPOINT:?MINIO_ENDPOINT is required}"
: "${MINIO_ACCESS_KEY:?MINIO_ACCESS_KEY is required}"
: "${MINIO_SECRET_KEY:?MINIO_SECRET_KEY is required}"
: "${MINIO_BUCKET:?MINIO_BUCKET is required}"

mc --config-dir /tmp/mc-config alias set local \
  "$MINIO_ENDPOINT" "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY" >/dev/null
mc --config-dir /tmp/mc-config ready local
mc --config-dir /tmp/mc-config mb --ignore-existing "local/$MINIO_BUCKET"
printf 'Bucket ready: %s\n' "$MINIO_BUCKET"
