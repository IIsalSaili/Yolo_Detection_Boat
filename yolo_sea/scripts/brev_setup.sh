#!/usr/bin/env bash
# Setup de l'environnement (venv + deps + dataset R2) sur une instance Brev.
#
# Le code est déjà sur disque à ce stade (cf. INSTRUCTIONS_BREV.md étape 2 —
# transfert par git archive + tar, le repo étant privé). Ce script ne clone
# rien : il suppose qu'il est lancé depuis la racine du repo extrait.
#
# Creds R2 Cloudflare à passer en variables d'env, JAMAIS commitées :
#   R2_ENDPOINT      ex. https://<accountid>.r2.cloudflarestorage.com
#   R2_ACCESS_KEY    Access Key ID
#   R2_SECRET_KEY    Secret Access Key
#   R2_BUCKET        nom du bucket
#   R2_DATASET_PATH  chemin du dataset combined dans le bucket (préfixe), défaut "combined"
#
# Usage :
#   cd ~/yolo_sea_homemade
#   R2_ENDPOINT=... R2_ACCESS_KEY=... R2_SECRET_KEY=... R2_BUCKET=... \
#     bash yolo_sea/scripts/brev_setup.sh
set -euo pipefail

: "${R2_ENDPOINT:?Renseigne R2_ENDPOINT}"
: "${R2_ACCESS_KEY:?Renseigne R2_ACCESS_KEY}"
: "${R2_SECRET_KEY:?Renseigne R2_SECRET_KEY}"
: "${R2_BUCKET:?Renseigne R2_BUCKET}"
R2_DATASET_PATH="${R2_DATASET_PATH:-combined}"

# ── 1. Venv ──────────────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m ensurepip --default-pip >/dev/null

# ── 2. PyTorch compatible avec le driver CUDA de l'instance ─────────────
# Indispensable sur certaines images H100/L40S : le torch par défaut de pip
# plante sur un mismatch driver/CUDA si on ne fixe pas l'index cu124.
python -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124

# ── 3. Ultralytics modifié (SESA/BiFPN/NWD/YOLO26-SEA) en editable ──────
python -m pip install -q -e .

# rclone pour pull le dataset depuis R2 (compatible S3)
if ! command -v rclone >/dev/null 2>&1; then
  curl -fsSL https://rclone.org/install.sh | sudo bash
fi

# ── 4. Images de test requises par check_amp() au démarrage du train ───
mkdir -p ultralytics/assets
[ -f ultralytics/assets/bus.jpg ] || curl -L -o ultralytics/assets/bus.jpg https://ultralytics.com/images/bus.jpg
[ -f ultralytics/assets/zelda.jpg ] || curl -L -o ultralytics/assets/zelda.jpg https://github.com/ultralytics/assets/raw/main/zelda.jpg

# ── 5. Config rclone R2 (provider S3 Cloudflare) ────────────────────────
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

# ── 6. Pull du dataset ───────────────────────────────────────────────────
mkdir -p datasets/combined
echo "Pull dataset depuis r2:${R2_BUCKET}/${R2_DATASET_PATH} …"
rclone copy "r2:${R2_BUCKET}/${R2_DATASET_PATH}" datasets/combined \
  --transfers 16 --checkers 16 --fast-list --progress

if [ ! -f datasets/combined/data.yaml ]; then
  echo "⚠️  data.yaml absent après pull — vérifie R2_DATASET_PATH (doit contenir data.yaml + images/labels)."
  ls -la datasets/combined
  exit 1
fi

echo "SETUP_DONE — venv activé dans ce shell. Pour le réactiver plus tard : source .venv/bin/activate"
