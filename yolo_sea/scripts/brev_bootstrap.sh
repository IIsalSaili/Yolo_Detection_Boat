#!/usr/bin/env bash
# Startup-script brev (SANS secret) : clone le code + install deps + rclone.
# Le pull R2 et le train sont lancés séparément via `brev exec` (creds injectés à ce moment-là).
set -euo pipefail

REPO_URL="https://github.com/IIsalSaili/yolo_sea_homemade.git"
BRANCH="feat/combined-dataset-single-cls"
WORKDIR="$HOME/yolo_sea_homemade"

# Code
if [ ! -d "$WORKDIR/.git" ]; then
  git clone --branch "$BRANCH" --depth 1 "$REPO_URL" "$WORKDIR"
else
  git -C "$WORKDIR" fetch origin "$BRANCH" && git -C "$WORKDIR" reset --hard "origin/$BRANCH"
fi
cd "$WORKDIR"

# Deps (ultralytics modifié en editable + rclone)
pip install -q -e . 2>&1 | tail -3 || pip install -q ultralytics
if ! command -v rclone >/dev/null 2>&1; then
  curl -fsSL https://rclone.org/install.sh | sudo bash
fi

echo "BOOTSTRAP_DONE"
