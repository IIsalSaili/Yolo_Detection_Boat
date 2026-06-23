# Déploiement sur Brev (Docker)

4 étapes : **créer l'instance → récupérer le code → builder l'image → lancer
le container avec les creds R2**.

L'image contient le code (SESA/BiFPN/NWD/YOLO26-SEA) et les libs. Le dataset
`combined` n'est PAS dans l'image — il est téléchargé depuis R2 (Cloudflare,
compatible S3) au démarrage du container, via `rclone`, à condition de passer
les bonnes variables d'environnement au `docker run` (étape 4).

---

## 1. Créer l'instance Brev

Choisir le mode **"VM"** (pas "Single Container") — le mode container n'a pas
de champ pour les variables d'environnement, donc pas de moyen propre de
passer les creds R2 sans les écrire en clair dans l'UI web.

```bash
brev create yolo-sea-gpu --gpu-name A100   # ou le GPU voulu
```

---

## 2. Récupérer le code sur l'instance

```bash
brev shell yolo-sea-gpu
git clone --branch feat/combined-dataset-single-cls --depth 1 \
  https://github.com/IIsalSaili/yolo_sea_homemade.git
cd yolo_sea_homemade
```

---

## 3. Builder l'image Docker

```bash
docker build -f docker/Dockerfile-sea-train -t yolo-sea-train .
```

---

## 4. Lancer le container — passer les creds R2 en argument

**Sans ces 4 variables, le dataset n'est pas téléchargé** (le script affiche
juste un avertissement et ouvre un bash vide). Elles se passent en `-e` au
`docker run`, jamais dans le Dockerfile :

```bash
docker run --gpus all -it \
  -e R2_ENDPOINT=https://<accountid>.r2.cloudflarestorage.com \
  -e R2_ACCESS_KEY=... \
  -e R2_SECRET_KEY=... \
  -e R2_BUCKET=... \
  yolo-sea-train
```

Ça pull le dataset dans `/workspace/datasets/combined` puis ouvre un bash
interactif.

Pour lancer directement l'entraînement (sans passer par le bash interactif),
ajouter la commande à la fin :

```bash
docker run --gpus all \
  -e R2_ENDPOINT=... -e R2_ACCESS_KEY=... -e R2_SECRET_KEY=... -e R2_BUCKET=... \
  -v "$PWD/runs:/workspace/runs" \
  yolo-sea-train \
  python yolo_sea/train_sea.py --dataset combined --single-cls --scale s --device 0
```

Le `-v "$PWD/runs:..."` est important : sans ça, les checkpoints/poids
disparaissent avec le container à la fin du run.
