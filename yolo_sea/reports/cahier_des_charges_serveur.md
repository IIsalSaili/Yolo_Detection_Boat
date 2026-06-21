# Cahier des charges — Serveur d'entraînement YOLO-SEA

> Projet : détection de navires à longue distance (YOLO26-SEA, base Ultralytics 8.4.60).
> Objet : spécifier la machine et l'environnement logiciel d'un **serveur dédié à l'entraînement** du modèle.
> Version : 1.0 — 12/06/2026

---

## 1. Contexte et objectif

Le modèle est aujourd'hui développé sur un poste **CPU-only** (torch `2.8.0+cpu`, pas de GPU CUDA),
ce qui rend l'entraînement réel impraticable (imgsz=1280, 300 epochs). Ce document définit les
besoins d'un serveur GPU permettant :

- l'entraînement complet (300 epochs) des variantes `yolo26{n,s}-sea` ;
- l'expérimentation (avec/sans NWD) sur les datasets du projet ;
- la marge pour les évolutions prévues (têtes multi-tâches, datasets externes volumineux type WUTDet 100k images).

**Le poste — son installation OS et logicielle — fait partie intégrante de la prestation.**

---

## 2. Caractérisation de la charge (workload)

| Paramètre | Valeur projet |
|-----------|---------------|
| Framework | PyTorch + Ultralytics 8.4.60 |
| Résolution d'entrée | **1280×1280** (gain principal sur petites cibles ; 640 en repli) |
| Modèles | yolo26n/s-sea (max **scale `s`** — contrainte projet) |
| Batch (réf. 8 Go) | 6 à 1280 ; 16 à 640 |
| Epochs | 300 (20 en mode `--quick`) |
| Optimizer | SGD, lr0=0.01, cosine, warmup 5 |
| Augmentations | mosaic, copy_paste=0.3, scale=0.9, HSV, rotation — **gourmandes en CPU** |
| Datasets actuels | fpp1 (1 776 img, 21 cl.), surveillance_marine (1 336), 6classfinal (~1 934) |
| Datasets cibles | + WUTDet (100k), ABOships (9,8k), MVDD13 (35k) → volume disque ↑ |
| Précision | FP16 (AMP) à l'entraînement, INT8 (QAT) visé à l'export |

**Implications dimensionnantes :**
- l'entraînement est **GPU-bound** (calcul) mais la chaîne d'augmentation à imgsz=1280 est **CPU-bound** sur le *dataloading* → il faut un ratio cœurs/GPU généreux ;
- la VRAM est le facteur limitant du batch → plus de VRAM = batchs plus grands = entraînement plus rapide/stable ;
- stockage **NVMe** indispensable dès qu'on passe aux datasets 10k–100k images.

---

## 3. Spécifications matérielles

Trois paliers. Le palier **Recommandé** est la cible par défaut.

### 3.1 GPU (composant critique)

| Palier | GPU | VRAM | Usage |
|--------|-----|------|-------|
| Minimum | 1× RTX 4070 Ti SUPER **16 Go** (ou RTX 3090 24 Go d'occasion) | 16–24 Go | imgsz=1280 batch 8–12, mono-modèle |
| **Recommandé** | **1× RTX 4090 24 Go** | **24 Go** | imgsz=1280 batch 16+, A/B confortable, meilleur €/perf |
| Confort / datacenter | 1× NVIDIA **L40S 48 Go** ou A100 40/80 Go | 48–80 Go | gros batchs, datasets 100k, multi-tâches, multi-utilisateurs |

- Architecture **Ampere ou plus récente** (compute capability ≥ 8.0) obligatoire pour AMP/INT8 efficaces.
- Multi-GPU (2× 4090) possible mais **non requis** : Ultralytics gère le DDP, mais le projet (scale `s`)
  n'en a pas besoin. Prévoir l'option seulement si entraînements parallèles.
- ⚠️ Éviter les GPU < 12 Go : forcent imgsz=640 et perdent l'intérêt longue distance.

### 3.2 CPU

| Palier | Spéc minimale |
|--------|---------------|
| Minimum | 8 cœurs / 16 threads (Ryzen 7 / Core i7 récent) |
| **Recommandé** | **16 cœurs / 32 threads** (Ryzen 9 7950X, Threadripper, ou Xeon/EPYC équivalent) |
| Confort | 24–32 cœurs (Threadripper PRO / EPYC) |

- Règle de dimensionnement : **≥ 8 cœurs physiques par GPU** pour saturer le dataloading à imgsz=1280
  avec mosaic+copy_paste. `workers=8` dans `train_sea.py` ; monter à 12–16 si CPU le permet.
- Fréquence mono-cœur correcte (≥ 3,5 GHz boost) utile pour le prétraitement.

### 3.3 Mémoire vive (RAM)

| Palier | RAM |
|--------|-----|
| Minimum | 32 Go DDR |
| **Recommandé** | **64 Go DDR** ECC de préférence |
| Confort | 128 Go (datasets 100k cachés en RAM, plusieurs jobs) |

- Règle : **≥ 2× la VRAM** et ≥ 2 Go par worker dataloader. 64 Go couvre 24 Go VRAM + cache datasets.

### 3.4 Stockage

| Rôle | Spéc |
|------|------|
| Système + datasets actifs | **SSD NVMe** 1 To (PCIe 4.0) — **obligatoire** |
| Archives datasets / runs | HDD ou SSD SATA 2–4 To (optionnel) |

- Datasets actuels ~2 Go, mais WUTDet/MVDD13 → dizaines de Go ; un NVMe évite que le disque
  devienne le goulet d'étranglement du dataloading.
- Prévoir l'espace pour les `runs/` (checkpoints `save_period=50`, plots, val) : ~1–5 Go par entraînement.

### 3.5 Alimentation / refroidissement / réseau

- Alimentation : **≥ 850 W** 80+ Gold (1000 W si RTX 4090 + CPU 16c) ; PSU de qualité (pics GPU).
- Refroidissement : boîtier ventilé / rack avec extraction ; une 4090 dissipe ~450 W en charge.
- Réseau : **1 Gbit/s minimum** (téléchargement datasets, poids COCO pretrain, accès distant).
  10 GbE si NAS de datasets partagé.

---

## 4. Système d'exploitation et environnement logiciel

### 4.1 Distribution

| Choix | Valeur |
|-------|--------|
| **OS recommandé** | **Ubuntu Server 22.04 LTS** (x86_64) |
| Alternative | Ubuntu 24.04 LTS, ou Debian 12 |
| À éviter | Windows Server (pilotes/CUDA possibles mais friction NCCL/DDP, perfs dataloader moindres) |

Rationale : Ubuntu LTS est la plateforme de référence PyTorch/NVIDIA — pilotes, CUDA, cuDNN et
roues PyTorch y sont testés en premier. Support jusqu'en 2027 (22.04) / 2029 (24.04).

### 4.2 Pile GPU / CUDA

| Composant | Version cible |
|-----------|---------------|
| Pilote NVIDIA | ≥ 550.x (compatible CUDA 12.4+) |
| CUDA Toolkit | **12.4** (fourni avec la roue PyTorch, pas besoin d'install système séparée) |
| cuDNN | celui embarqué dans la roue PyTorch |
| NVIDIA Container Toolkit | si déploiement Docker (recommandé) |

### 4.3 Environnement Python

| Composant | Version |
|-----------|---------|
| Python | **3.11** (ou 3.10) |
| Gestionnaire d'env | `venv` ou `conda`/`mamba` |
| PyTorch | **build CUDA** : `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| Ultralytics | déjà vendu dans le repo (8.4.60) — installer en editable : `pip install -e .` |
| Dépendances projet | `polars` (lecture results.csv), + requirements Ultralytics |

> ⚠️ Ne **pas** réinstaller la roue `+cpu` : c'est la cause du blocage actuel. Vérifier après install :
> `python -c "import torch; print(torch.cuda.is_available())"` doit afficher **True**.

### 4.4 Conteneurisation (optionnel mais conseillé)

Image de base `nvidia/cuda:12.4.x-cudnn-runtime-ubuntu22.04` ou `pytorch/pytorch:2.x-cuda12.4-cudnn9-runtime`,
+ le repo monté en volume. Garantit la reproductibilité et isole les versions CUDA.

---

## 5. Configuration et accès

- **Accès distant** : SSH (clé publique, pas de mot de passe), + tunnel pour TensorBoard/visualisation si besoin.
- **Comptes** : 1 utilisateur non-root dédié au projet ; `sudo` réservé à l'admin.
- **Settings Ultralytics** : configurer `datasets_dir` et `runs_dir` vers des chemins absolus du serveur
  (⚠️ historique : pointaient vers `Desktop\modif_yolo` sur le poste Windows — à réinitialiser).
- **Datasets** : déposer `fpp1/`, `surveillance_marine/`, `6classfinal/` sous un dossier dédié ;
  `data.yaml` sans clé `path:` (résolution sur le dossier du yaml — déjà appliqué à fpp1).

---

## 6. Estimation de durée d'entraînement (ordre de grandeur)

Indicatif, dataset fpp1 (~1 600 img train), 300 epochs, imgsz=1280, yolo26s-sea :

| Configuration | Durée estimée 300 epochs |
|---------------|--------------------------|
| CPU seul (poste actuel) | **impraticable** (semaines) — d'où ce serveur |
| RTX 4070 Ti S 16 Go | ~10–16 h |
| **RTX 4090 24 Go** | **~5–9 h** |
| L40S / A100 | ~3–6 h |

À ajuster selon batch effectif et augmentations. Le mode `--quick --imgsz 640` valide le pipeline en minutes.

---

## 7. Sécurité, sauvegarde, exploitation

- **Sauvegarde** : `runs/*/weights/best.pt` et `last.pt` répliqués hors machine (NAS/cloud) après chaque run.
- **Monitoring** : `nvidia-smi`/`nvtop` pour la charge GPU, surveillance température/alim.
- **Onduleur (UPS)** recommandé si on-prem : un entraînement de plusieurs heures perdu sur coupure secteur.
- **Versionnage** : le code (modifs Ultralytics, configs, scripts) reste sous git ; datasets/poids hors git
  (déjà gitignorés).

---

## 8. Alternative cloud (si pas d'achat matériel)

Location GPU à l'heure, équivalent fonctionnel :

| Fournisseur (type) | Instance type | Note |
|--------------------|---------------|------|
| Vast.ai / RunPod | RTX 4090 24 Go | le moins cher, idéal expérimentation |
| Lambda / Paperspace | A100 40 Go | si besoin de gros batchs |
| AWS / GCP / Azure | g5 (A10G) / a2 (A100) | si contrainte entreprise/conformité |

Avantages : pas d'investissement, scaling à la demande. Inconvénients : coût récurrent, transfert
des datasets, gestion des sessions. **Recommandé pour démarrer / valider avant un achat on-prem.**

---

## 9. Synthèse — configuration recommandée

| Poste | Spécification cible |
|-------|---------------------|
| **OS** | Ubuntu Server 22.04 LTS x86_64 |
| **GPU** | 1× NVIDIA RTX 4090 24 Go (Ampere+ ; min 16 Go) |
| **CPU** | 16 cœurs / 32 threads (≥ 8 cœurs/GPU) |
| **RAM** | 64 Go DDR (ECC souhaité) |
| **Stockage** | NVMe PCIe 4.0 1 To (système + datasets) + 2 To archives |
| **Alim.** | 1000 W 80+ Gold + UPS |
| **CUDA/PyTorch** | pilote ≥ 550, PyTorch build cu124, Python 3.11 |
| **Réseau** | 1 GbE (10 GbE si NAS), accès SSH par clé |

---

## 10. Critères d'acceptation

1. `nvidia-smi` détecte le(s) GPU ; pilote ≥ 550 installé.
2. `python -c "import torch; print(torch.cuda.is_available())"` → **True**, et `torch.cuda.get_device_name(0)` correct.
3. `python train_sea.py --dataset fpp1 --quick --imgsz 640` se termine **sans erreur** et écrit un `best.pt`.
4. `python train_sea.py --dataset fpp1` (gen 26, 1280) tourne sur GPU avec `Device : cuda:0` au démarrage.
5. Utilisation GPU > 80 % pendant l'entraînement (vérifiée à `nvtop`), pas de goulet CPU/disque.
6. Accès SSH par clé fonctionnel, sauvegarde des poids opérationnelle.
