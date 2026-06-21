# AVANCEMENT — Projet YOLO-SEA / Détection navires longue distance

> Fichier de suivi. Voir `CLAUDE.md` pour la spécification complète.
> Base : Ultralytics YOLO `8.4.60`. NMS dans `ultralytics/utils/nms.py`.

---

## État global

| # | Phase | Statut |
|---|-------|--------|
| 1 | Architecture YOLO-SEA (SESA + BiFPN + Soft-NMS) | ✅ fait |
| 2 | Dataset surveillance_marine + entraînement | ✅ fait |
| 3 | Script détection avec affichage coloré | ✅ fait |
| 4 | Workflow annotation Label Studio | ✅ fait |
| 5 | Guide de reconnaissance navires (HTML) | ✅ fait |
| 6 | Architecture multi-têtes (décision) | 🔧 en cours |
| 7 | Tête classification civil/militaire | ⏳ à faire |
| 8 | Tête type + nationalité | ⏳ à faire |
| 9 | Tête keypoints (orientation) | ⏳ à faire |
| 10 | Migration YOLO26-SEA (NMS-free, P2, NWD) | ✅ fait |

Légende : ⏳ à faire / 🔧 en cours / ✅ fait

**Contrainte matériel** : RTX 4060 Ti 8 Go → max `yolov8s`, batch=8 à imgsz=1280.

---

## Phase 1 — Architecture YOLO-SEA (✅)

### SESA — backbone
- `ultralytics/nn/modules/attention.py` : `SENetV2`, `SimAM`, `SESA`
- `SimAM` = 0 paramètre (attention spatiale sans paramètre)
- `SENetV2` = squeeze multi-branches + excitation (gate ×)
- `SESA` = SENetV2 (canaux) → SimAM (spatial), canaux préservés
- Enregistré dans `nn/modules/__init__.py` et `nn/tasks.py`

### BiFPN — neck
- `BiFPNAdd2` / `BiFPNAdd3` dans `nn/modules/block.py`
- Fast normalized fusion : poids appris, relu, normalisation/(Σw+ε), ε=1e-4
- Neck bidirectionnel (top-down + bottom-up), skip P4

### Config modèle
- `cfg/models/v8/yolov8-sea.yaml` : backbone + SESA (idx 9, avant SPPF) + BiFPN P3/P4/P5
- Params scale n (nc=6) ≈ 2,09M. Forward 640² → `(1,84,8400)` OK.

### Soft-NMS
- `ultralytics/utils/nms.py` : flags `USE_SOFT_NMS` / `SOFT_NMS_SIGMA` / `SOFT_NMS_METHOD`
- Decay gaussien `exp(-iou²/σ)` ou linéaire
- Désactivé par défaut. Activation : `nms.USE_SOFT_NMS = True`

---

## Phase 2 — Dataset et entraînement (✅)

### Datasets utilisés
| Dataset | Images | Classes | Source |
|---------|--------|---------|--------|
| `fpp1/` | 1 598 train / 178 val | **21** (10 mil OTAN : carrier, cruiser, destroyer, frigate, corvette, submarine, amphibious, mcm, patrol_mil, aor + 10 civ + unknown) | LabelDeZero — export 12/06/2026 |
| `surveillance_marine/` | 919 train / 254 val / 163 test | 7 (boat, buoy, kayak, sail-boat, speed-boat, vessel, war-ship) | Roboflow — marine-surveillance v9 |
| `6classfinal/` | ~1 934 | 6 (a, buoy, container, cruise, fish-b, warship) | Roboflow — 6_class_final v1 |

> `fpp1` est le dataset le plus riche (taxonomie projet complète) → câblé dans `yolo_sea/train_sea.py`
> (`--dataset fpp1`). `data.yaml` sans clé `path:` (résolution sur le dossier du yaml, CWD-indépendant).

### Script d'entraînement : `yolo_sea/train_sea.py`
```
python yolo_sea/train_sea.py                        # surveillance, 300 epochs, imgsz=1280
python yolo_sea/train_sea.py --dataset 6classfinal
python yolo_sea/train_sea.py --imgsz 640
python yolo_sea/train_sea.py --quick                # 20 epochs test
python yolo_sea/train_sea.py --baseline             # YOLOv8n baseline
python yolo_sea/train_sea.py --scale s
python yolo_sea/train_sea.py --resume <last.pt>
```

Hyperparamètres clés (optimisés longue distance) :
- `imgsz=1280` — gain principal sur petites cibles
- `optimizer=SGD, lr0=0.01, momentum=0.937` (0.01 du papier = coquille)
- `box=10.0, cls=0.3` — priorité localisation sur classification
- `copy_paste=0.3, scale=0.9` — augmentation petites cibles
- `cos_lr=True, warmup_epochs=5`
- Partial COCO pretrain : `model.load("yolov8n.pt")` (backbone Conv/C2f layers 0–8)

### Script détection : `yolo_sea/detect_sea.py`
```
python yolo_sea/detect_sea.py image.jpg
python yolo_sea/detect_sea.py 0                     # webcam
python yolo_sea/detect_sea.py video.mp4 --save out.mp4
python yolo_sea/detect_sea.py image.jpg --soft-nms
```
Supporte les deux datasets avec couleurs par classe. Poids par défaut configurables dans `DATASET_CFG`.

---

## Phase 3 — Workflow annotation Label Studio (✅)

### Fichiers
| Fichier | Rôle |
|---------|------|
| `annotations/label_studio_template.xml` | Template interface Label Studio |
| `annotations/guide_annotateurs.md` | Guide pour les annotateurs |

### Interface Label Studio — layout
```
[    IMAGE    ] [ Détection  ] [ Nationalité ] [ Keypoints  ]
               [ Statut     ]               [ Conditions ]
               [ Type       ]
[ BARRE RACCOURCIS ]
```

### Classes annotées
- **Détection** : `ship` (classe unique, toujours)
- **Statut** : civil / militaire / inconnu (perRegion)
- **Type militaire** : 10 types OTAN — CV, CG, DD, FF, FS, PC, SS, LHA, AO, MCM (perRegion)
- **Type civil** : `civil` classe unique (classification fine = tête dédiée future)
- **Nationalité** : 19 pays (perRegion)
- **Keypoints** : proue, poupe, mât, passerelle, babord, tribord (si navire > ~50px)
- **Météo** : dégagé, brume, brouillard, pluie, nuit, contre-jour (niveau image)
- **Distance** : proche, loin, très loin (niveau image)

### Import dans Label Studio
Settings → Labeling Interface → Code → coller le contenu de `label_studio_template.xml`

---

## Phase 4 — Guide de reconnaissance navires (✅)

### Fichiers
| Fichier | Rôle |
|---------|------|
| `annotations/guide_navires_guerre_v3.html` | Guide interactif HTML — ~170 classes en service |
| `annotations/img_cache.js` | Cache images local (généré par script) |
| `annotations/download_images.py` | Télécharge images Wikipedia |
| `annotations/normalize_images.py` | Normalise noms de fichiers manuels |

### Usage
```
cd annotations
python download_images.py        # télécharge images manquantes + génère img_cache.js
python normalize_images.py       # dry-run
python normalize_images.py --apply  # renomme les images téléchargées manuellement
```

### Fonctionnalités du guide HTML
- Galerie avec filtres par type OTAN + nationalité (drapeaux)
- Recherche texte
- Modal avec détails + lien Wikipedia
- Images : affichage local immédiat (img_cache.js), fallback Wikipedia au clic modal
- ~170 classes couvertes (porte-avions, destroyers, frégates, corvettes, sous-marins, amphibies)

---

## Phase 5 — Architecture multi-têtes (décision prise, implémentation future)

### Décision architecturale
Le modèle sera multi-tâche avec têtes activées progressivement selon la distance :

| Distance | Tête active |
|----------|------------|
| > 300m (< 15px) | Détection uniquement |
| 150–300m | + Civil / Militaire |
| 50–150m | + Type de navire |
| < 50m | + Nationalité + Keypoints |

### Têtes prévues
1. **Détection** — classe unique `ship` ✅ (déjà opérationnel)
2. **Civil/Militaire** — binaire + inconnu (prochaine étape)
3. **Type OTAN** — 10 classes militaires + civil
4. **Nationalité** — classification par marine
5. **Keypoints** — proue/poupe/mât/passerelle/babord/tribord → cap déduit

### Implémentation
- Chaque tête greffée sur les features du détecteur (pas dans le backbone)
- Loss maskée pour les têtes sans label dans le batch courant
- Cascade conditionnelle à l'inférence (seuil sur taille de boîte en pixels)
- Référence : YOLOv8-pose pour la plomberie keypoints

---

## Phase 6 — Migration YOLO26-SEA (✅)

> Issue de l'étude d'architecture `yolo_sea/reports/recherche_architecture.md` (état de l'art 2024–2026).
> Le modèle par défaut du projet est désormais **YOLO26-SEA** ; l'architecture v8 reste disponible (`--gen v8`).

### Architecture — `cfg/models/26/yolo26-sea.yaml`
- **Backbone YOLO26** (C3k2 / SPPF / C2PSA) + **SESA** (idx 9, avant SPPF — inchangé vs v8-sea)
- **Neck BiFPN P2–P5** : niveau **P2/4 ajouté** pour les très petites cibles (navires lointains),
  fusion pondérée `BiFPNAdd2/3`, skips sur les niveaux intermédiaires P3/P4. Largeur commune 256.
- **Tête Detect end-to-end (NMS-free)** + `reg_max=1` (pas de DFL) → export ONNX/TensorRT et
  quantification INT8 simplifiés. Soft-NMS sans objet sur cette génération (reste utile pour v8).
- Params : **n = 1,94M** (nc=80) / 1,86M (nc=7) — **s = 7,45M**. Strides `[4, 8, 16, 32]`.
  Forward 640² → `(1, 300, 6)` directement (pas de post-traitement).
- Sélection du scale par nom de fichier : `yolo26s-sea.yaml` → charge `yolo26-sea.yaml` en scale `s`.

### NWD — box-similarity pour très petites cibles (opt-in)
- `utils/metrics.py` : `wasserstein_similarity()` + flags `USE_NWD` / `NWD_RATIO` / `NWD_C`
  (NWD, arXiv:2110.13389 — l'IoU est quasi binaire pour une boîte de quelques pixels).
- Branchée comme **blend CIoU/NWD** dans `BboxLoss` (`utils/loss.py`, rescale grille→pixels par
  stride) **et** dans `TaskAlignedAssigner.iou_calculation` (`utils/tal.py`, déjà en pixels).
- Off par défaut → zéro impact sur les modèles stock. Effet **entraînement uniquement**
  (assignation + loss), l'inférence est inchangée. Activation : `python yolo_sea/train_sea.py --nwd`.

### Scripts
- `yolo_sea/train_sea.py` : `--gen {26,v8}` (défaut **26**), `--nwd` / `--nwd-ratio` / `--nwd-c`,
  batch auto 6 à imgsz=1280 pour gen 26 (la tête P2 quadruple les anchors du niveau fin).
  **Correction** : `--scale s` construisait en réalité l'échelle `n` sur le yaml sea (le scale
  n'était pas transmis) — corrigé en injectant le scale dans le nom du yaml.
- `yolo_sea/detect_sea.py` : `--soft-nms` détecté comme sans effet sur un modèle end2end (message explicite).

### Validation (smoke test)
- Build n/s OK, forward OK, 1 epoch d'entraînement complète avec NWD ON (loss/assigner/val/save OK).
- Dépendance ajoutée : `polars` (requis par le trainer 8.4.x pour lire results.csv).
- ⚠️ Settings Ultralytics globaux : `runs_dir` pointe vers `Desktop\modif_yolo` (ancien projet).
  `yolo_sea/train_sea.py` utilise des chemins absolus donc non affecté ; éviter les `project=` relatifs.

### Prochaines étapes issues de la recherche (voir `yolo_sea/reports/recherche_architecture.md`)
1. Comparaison A/B : `--gen 26` vs `--gen v8`, avec/sans `--nwd`, sur surveillance_marine.
2. Tête classification hiérarchique conditionnelle (civil/militaire → type OTAN) avec abstention.
3. Tête keypoints (proue/poupe → cap) + gating par taille de boîte (cf. Phase 5).
4. QAT INT8 sur le checkpoint final ; pistes recherche : prior horizon, module temporel.

---

## Datasets supplémentaires identifiés (pour enrichir l'entraînement)

| Dataset | Images | Pertinence | Lien |
|---------|--------|------------|------|
| WUTDet | 100 576 | Très haute — petites cibles, 1 classe ship, conditions adverses | github.com/MAPGroup/WUTDet |
| ABOships | 9 880 | Haute — classe `militaryship`, niveau de la mer | zenodo.org/records/4736931 |
| MVDD13 | 35 474 | Haute — warship + submarine explicites, mixte | github.com/yyuanwang1010/MVDD13 |
| Navy Ship (Roboflow) | 2 125 | Moyenne — navires militaires photographiques | universe.roboflow.com/navy-ip6vd |
| HRSC2016 | 1 680 | Satellite — OBB, types militaires précis | kaggle.com/datasets/guofeng/hrsc2016 |
| ShipRSImageNet | 3 435 | Satellite — 50 classes dont militaires | github.com/zzndream/ShipRSImageNet |

---

## Reprise rapide (si crash)
1. Lire ce fichier + `CLAUDE.md`
2. Vérifier le statut de chaque phase
3. Pour relancer l'entraînement : `python yolo_sea/train_sea.py --resume runs/sea/.../last.pt`
4. Pour l'annotation : ouvrir Label Studio, importer le template XML
