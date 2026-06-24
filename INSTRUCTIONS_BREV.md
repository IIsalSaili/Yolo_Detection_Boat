# Déploiement sur Brev (venv, sans Docker)

3 étapes : **créer l'instance → envoyer le code → setup venv + dataset →
lancer le train**.

Pas de build Docker : on transfère juste le code source (sans `.venv`) et on
installe les dépendances directement dans un venv sur l'instance — un build
d'image n'apporte rien ici puisqu'il faut de toute façon transférer tout le
repo comme contexte. Le dataset `combined` n'est pas transféré : il est tiré
depuis R2 (Cloudflare, compatible S3) via `rclone` à l'étape 3.

---

## 1. Créer l'instance Brev

```bash
brev create yolo-sea-gpu --gpu-name A100   # ou le GPU voulu
```

---

## 2. Envoyer le code sur l'instance

Le repo GitHub est **privé** : un `git clone` échouerait sur l'instance (pas
de creds Git configurées dessus). On empaquette donc uniquement les fichiers
trackés par Git avec `git archive` (ça exclut `.venv/`, `datasets/`, `runs/`
et tout fichier non commité), puis on envoie ce tarball :

```bash
# depuis ta machine locale, à la racine du repo
git archive --format=tar.gz -o /tmp/yolo_sea_homemade.tar.gz HEAD
brev copy /tmp/yolo_sea_homemade.tar.gz yolo-sea-gpu:~/yolo_sea_homemade.tar.gz

brev shell yolo-sea-gpu
mkdir -p yolo_sea_homemade && tar -xzf yolo_sea_homemade.tar.gz -C yolo_sea_homemade
cd yolo_sea_homemade
```

---

## 3. Setup (venv + deps + dataset)

`brev_setup.sh` crée le venv, installe PyTorch (index CUDA adapté au driver
de l'instance) et le package `ultralytics` modifié en editable, puis pull le
dataset depuis R2. **Sans les 4 variables R2, le script s'arrête en erreur**
avant de toucher au dataset :

```bash
R2_ENDPOINT=https://<accountid>.r2.cloudflarestorage.com \
R2_ACCESS_KEY=... \
R2_SECRET_KEY=... \
R2_BUCKET=... \
bash yolo_sea/scripts/brev_setup.sh
```

À la fin du script, le venv est activé dans le shell courant. Dans une
nouvelle session, le réactiver avec :

```bash
source .venv/bin/activate
```

---

## 4. Lancer l'entraînement

```bash
python yolo_sea/train_sea.py --dataset combined --single-cls --scale s --device 0
```

Adapter `--batch` à la VRAM de l'instance (défaut 512 dans `train_sea.py`,
bien trop haut pour un GPU 8–24 Go — ex. `--batch 24` sur une L40S 48 Go,
`--batch 6` sur une 4060 Ti 8 Go).

Penser à rediriger les logs et à sauvegarder `runs/` ailleurs que sur le
disque éphémère de l'instance si elle doit être détruite après le run.
