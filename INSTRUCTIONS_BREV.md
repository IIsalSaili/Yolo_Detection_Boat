# Guide de déploiement et configuration d'une instance Brev

Ce fichier regroupe toutes les commandes nécessaires pour transférer votre code local vers une instance Brev, configurer l'environnement Python et lancer l'entraînement.

---

## 1. Depuis votre machine locale (Transfert du code)

Avant de transférer, assurez-vous que votre instance est démarrée (`RUNNING` / `READY`) et rafraîchissez votre configuration SSH locale :
```bash
brev refresh
```

Utilisez `rsync` pour copier le dossier du projet vers l'instance Brev (remplacez `NOM_INSTANCE` par le nom de votre instance, ex. `damaged-black-elephant`) :
```bash
rsync -avz --progress --exclude='.venv' /home/dromsis/Documents/yolo_sea_homemade NOM_INSTANCE:/home/shadeform/
```

---

## 2. Sur l'instance Brev (Configuration de l'environnement)

Connectez-vous à l'instance (via `brev shell NOM_INSTANCE` ou directement depuis le terminal distant) et lancez ces commandes dans l'ordre :

### Étape A : Se placer dans le dossier du projet
```bash
cd ~/yolo_sea_homemade
```

### Étape B : Activer l'environnement virtuel et installer Pip (si manquant)
```bash
source ~/.venv/bin/activate
python -m ensurepip --default-pip
```

### Étape C : Installer PyTorch compatible avec le pilote CUDA de l'instance
*Note : Indispensable pour éviter le crash d'incompatibilité de driver sur les instances de type H100/L40S.*
```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
```

### Étape D : Installer le package Ultralytics modifié en mode éditable
```bash
python -m pip install -e .
```

### Étape E : Télécharger les images de test d'origine (requises par l'AMP Check)
```bash
mkdir -p ultralytics/assets
curl -L -o ultralytics/assets/bus.jpg https://ultralytics.com/images/bus.jpg
curl -L -o ultralytics/assets/zelda.jpg https://github.com/ultralytics/assets/raw/main/zelda.jpg
```

---

## 3. Lancer l'entraînement

Une fois la configuration terminée, vous pouvez démarrer l'entraînement (assurez-vous d'avoir activé le `.venv` au préalable) :

```bash
python yolo_sea/train_sea.py --dataset combined --single-cls --batch 64 --device 0 --nwd --epochs 20
```
