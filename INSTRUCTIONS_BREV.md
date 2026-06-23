# Guide de déploiement sur une instance Brev (Docker)

Ce fichier regroupe les commandes pour builder l'image Docker d'entraînement
YOLO26-SEA **directement sur l'instance Brev** (GPU), puis la lancer. Pas de
push sur un registre : le réseau d'une instance cloud est généralement bien
plus rapide que celui d'une connexion résidentielle, donc on évite le double
goulot téléchargement (image de base ~4,3 Go) + upload (image finale ~5 Go)
depuis chez soi.

L'image embarque le code du repo (modifications SESA/BiFPN/NWD/YOLO26-SEA) et
toutes les libs nécessaires. Le dataset `combined` est pull depuis R2
(Cloudflare, compatible S3) au démarrage du container, via `rclone`.

---

## 1. Sur l'instance Brev : récupérer le code

Connectez-vous à l'instance (`brev shell NOM_INSTANCE` ou terminal distant) :

```bash
git clone --branch feat/combined-dataset-single-cls --depth 1 \
  https://github.com/IIsalSaili/yolo_sea_homemade.git
cd yolo_sea_homemade
```

(Si le repo est déjà cloné : `git fetch origin feat/combined-dataset-single-cls && git checkout feat/combined-dataset-single-cls && git pull`.)

---

## 2. Build de l'image (sur l'instance)

```bash
docker build -f docker/Dockerfile-sea-train -t yolo-sea-train .
```

---

## 3. Lancer le container

Les creds R2 sont passés en variables d'env au `docker run` — jamais
commités ni baked dans l'image :

```bash
docker run --gpus all -it \
  -e R2_ENDPOINT=https://<accountid>.r2.cloudflarestorage.com \
  -e R2_ACCESS_KEY=... \
  -e R2_SECRET_KEY=... \
  -e R2_BUCKET=... \
  -e R2_DATASET_PATH=combined \
  yolo-sea-train
```

L'entrypoint (`yolo_sea/scripts/docker_entrypoint.sh`) pull le dataset
`combined` depuis R2 dans `/workspace/datasets/combined`, puis ouvre un bash
interactif (aucune commande de train n'est lancée automatiquement).

Pour enchaîner directement sur l'entraînement, ajouter la commande à la fin :

```bash
docker run --gpus all \
  -e R2_ENDPOINT=... -e R2_ACCESS_KEY=... -e R2_SECRET_KEY=... -e R2_BUCKET=... \
  -v "$PWD/runs:/workspace/runs" \
  yolo-sea-train \
  python yolo_sea/train_sea.py --dataset combined --single-cls --scale s --device 0
```

Penser à monter `runs/` en volume (`-v`) si les poids/checkpoints doivent
survivre à la suppression du container.

---

## Alternative : build local + push registre

Si l'instance Brev n'a pas accès à `git clone` (réseau restreint) ou si vous
préférez réutiliser la même image sur plusieurs instances sans rebuild :

```bash
docker build -f docker/Dockerfile-sea-train -t yolo-sea-train .
docker tag yolo-sea-train <registre>/<user>/yolo-sea-train:latest
docker push <registre>/<user>/yolo-sea-train:latest
# puis sur l'instance Brev :
docker run --gpus all -it -e R2_ENDPOINT=... -e R2_ACCESS_KEY=... -e R2_SECRET_KEY=... -e R2_BUCKET=... \
  <registre>/<user>/yolo-sea-train:latest
```
