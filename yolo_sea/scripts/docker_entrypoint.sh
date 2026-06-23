#!/usr/bin/env bash
# Entrypoint de l'image Docker yolo-sea-train.
#
# Au démarrage du container : pull le dataset "combined" depuis R2 (Cloudflare,
# compatible S3) via rclone, puis exécute la commande passée à `docker run`
# (ou ouvre un bash interactif si aucune commande n'est fournie).
#
# Creds R2 — à passer en variables d'env au `docker run` / config Brev,
# JAMAIS commitées ni baked dans l'image :
#   R2_ENDPOINT      ex. https://<accountid>.r2.cloudflarestorage.com
#   R2_ACCESS_KEY    Access Key ID
#   R2_SECRET_KEY    Secret Access Key
#   R2_BUCKET        nom du bucket
#   R2_DATASET_PATH  chemin du dataset combined dans le bucket (préfixe), défaut "combined"
#
# Exemples :
#   docker run --gpus all -e R2_ENDPOINT=... -e R2_ACCESS_KEY=... -e R2_SECRET_KEY=... \
#     -e R2_BUCKET=... yolo-sea-train
#     → pull dataset puis bash interactif
#
#   docker run --gpus all -e R2_ENDPOINT=... -e R2_ACCESS_KEY=... -e R2_SECRET_KEY=... \
#     -e R2_BUCKET=... yolo-sea-train python yolo_sea/train_sea.py --dataset combined --single-cls --device 0
#     → pull dataset puis lance directement le train
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
R2_DATASET_PATH="${R2_DATASET_PATH:-combined}"

if [ -n "${R2_ENDPOINT:-}" ] && [ -n "${R2_ACCESS_KEY:-}" ] && [ -n "${R2_SECRET_KEY:-}" ] && [ -n "${R2_BUCKET:-}" ]; then
  mkdir -p "$HOME/.config/rclone"
  cat > "$HOME/.config/rclone/rclone.conf" <<EOF
[r2]
type = s3
provider = Cloudflare
access_key_id = ${R2_ACCESS_KEY}
secret_access_key = ${R2_SECRET_KEY}
endpoint = ${R2_ENDPOINT}
acl = private
EOF

  mkdir -p "$WORKDIR/datasets/combined"
  echo "Pull dataset depuis r2:${R2_BUCKET}/${R2_DATASET_PATH} …"
  rclone copy "r2:${R2_BUCKET}/${R2_DATASET_PATH}" "$WORKDIR/datasets/combined" \
    --transfers 16 --checkers 16 --fast-list --progress

  if [ ! -f "$WORKDIR/datasets/combined/data.yaml" ]; then
    echo "⚠️  data.yaml absent après pull — vérifie R2_DATASET_PATH (doit contenir data.yaml + images/labels)."
    ls -la "$WORKDIR/datasets/combined"
    exit 1
  fi
else
  echo "⚠️  Variables R2_ENDPOINT/R2_ACCESS_KEY/R2_SECRET_KEY/R2_BUCKET absentes — dataset non téléchargé."
  echo "   Passe-les en -e au 'docker run' pour le pull automatique."
fi

if [ "$#" -eq 0 ]; then
  exec bash
else
  exec "$@"
fi
