#!/usr/bin/env bash
# Setup + lancement du train YOLO-SEA longue distance sur une instance brev L40S.
#
# À renseigner (creds R2 Cloudflare) — passés en variables d'env, JAMAIS commités :
#   R2_ENDPOINT      ex. https://<accountid>.r2.cloudflarestorage.com
#   R2_ACCESS_KEY    Access Key ID
#   R2_SECRET_KEY    Secret Access Key
#   R2_BUCKET        nom du bucket
#   R2_DATASET_PATH  chemin du dataset combined dans le bucket (préfixe), ex. "datasets/combined"
#
# Usage sur l'instance :
#   R2_ENDPOINT=... R2_ACCESS_KEY=... R2_SECRET_KEY=... \
#   R2_BUCKET=... R2_DATASET_PATH=... bash brev_setup.sh
set -euo pipefail

REPO_URL="https://github.com/IIsalSaili/yolo_sea_homemade.git"
BRANCH="feat/combined-dataset-single-cls"
WORKDIR="$HOME/yolo_sea_homemade"

# ── 0. Vérif des creds R2 ────────────────────────────────────────────────
: "${R2_ENDPOINT:?Renseigne R2_ENDPOINT}"
: "${R2_ACCESS_KEY:?Renseigne R2_ACCESS_KEY}"
: "${R2_SECRET_KEY:?Renseigne R2_SECRET_KEY}"
: "${R2_BUCKET:?Renseigne R2_BUCKET}"
R2_DATASET_PATH="${R2_DATASET_PATH:-datasets/combined}"

# ── 1. Code ──────────────────────────────────────────────────────────────
if [ ! -d "$WORKDIR/.git" ]; then
  git clone --branch "$BRANCH" --depth 1 "$REPO_URL" "$WORKDIR"
else
  git -C "$WORKDIR" fetch origin "$BRANCH" && git -C "$WORKDIR" checkout "$BRANCH" && git -C "$WORKDIR" pull
fi
cd "$WORKDIR"

# ── 2. Dépendances ───────────────────────────────────────────────────────
# ultralytics en mode editable (le repo contient une version modifiée : SESA, BiFPN, NWD…)
pip install -q -e . || pip install -q -r requirements.txt || true
# rclone pour pull le dataset depuis R2 (compatible S3)
if ! command -v rclone >/dev/null 2>&1; then
  curl -fsSL https://rclone.org/install.sh | sudo bash
fi

# ── 3. Config rclone R2 (provider S3 Cloudflare) ─────────────────────────
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

# ── 4. Pull du dataset ───────────────────────────────────────────────────
mkdir -p "$WORKDIR/datasets/combined"
echo "Pull dataset depuis r2:${R2_BUCKET}/${R2_DATASET_PATH} …"
rclone copy "r2:${R2_BUCKET}/${R2_DATASET_PATH}" "$WORKDIR/datasets/combined" \
  --transfers 16 --checkers 16 --fast-list --progress

if [ ! -f "$WORKDIR/datasets/combined/data.yaml" ]; then
  echo "⚠️  data.yaml absent après pull — vérifie R2_DATASET_PATH (doit contenir data.yaml + images/labels)."
  ls -la "$WORKDIR/datasets/combined"
  exit 1
fi

# ── 5. Train ─────────────────────────────────────────────────────────────
# L40S = 48 Go VRAM → on monte le batch (le script défaut à 6 pour une 4060 Ti 8 Go).
# imgsz=1280, YOLO26-SEA, mono-classe (dataset combined annoté en classe unknown).
echo "Lancement du train …"
python train_sea.py --dataset combined --single-cls --batch 24 --device 0 2>&1 \
  | tee "$WORKDIR/train_combined.log"
