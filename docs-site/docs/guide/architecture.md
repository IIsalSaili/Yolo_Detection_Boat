# Architecture globale

Fichier de configuration : `ultralytics/cfg/models/26/yolo26-sea.yaml`

```
Input 640×640 (ou 1280×1280 pour cibles très lointaines)
    │
    ▼
[BACKBONE — YOLO26 + SESA]
  Conv → Conv → C3k2 (P2) → Conv → C3k2 (P3) → Conv → C3k2 (P4)
       → Conv → C3k2 → SESA → SPPF → C2PSA (P5)
    │
    ▼
[NECK — BiFPN P2-P5]
  Projection 1×1 → 256 canaux (P2_in, P3_in, P4_in, P5_in)
  Top-down   : P5_in → P4_td → P3_td → P2_out          (BiFPNAdd2)
  Bottom-up  : P2_out → P3_out → P4_out → P5_out        (BiFPNAdd3, skips)
    │
    ▼
[HEAD]
  Detect(P2, P3, P4, P5) — end2end, reg_max=1 (NMS-free)
    │
    ▼
[POST-TRAITEMENT — optionnel]
  Soft-NMS (si end2end désactivé / modèle non-e2e)
```

## Échelles supportées

Le fichier `yolo26-sea.yaml` déclare les 5 scales standard (`n`, `s`, `m`, `l`, `x`), on se cantonne ici à s et n, dans un soucis de limitation mémoire pour l'usage dans un système embarqué. 


```yaml
scales: # 'n' and 's' are the supported targets for this project (hardware constraint)
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]
```

## Points clés vs YOLO26 stock

- **Backbone** : identique jusqu'au dernier `C3k2` (index 8), puis insertion de `SESA` (index 9) avant
  `SPPF` (index 10) et `C2PSA` (index 11).
- **Neck** : remplace le PAN classique par un **BiFPN à 4 niveaux (P2-P5)** avec fusion pondérée
  apprenante (`BiFPNAdd2` / `BiFPNAdd3`), voir [BiFPN](/guide/bifpn).
- **Tête de détection** : `Detect` standard end-to-end sur les 4 niveaux de sortie du neck
  (P2_out, P3_out, P4_out, P5_out).
- **Post-traitement** : `Soft-NMS` disponible dans `ultralytics/utils/nms.py` pour les variantes
  non end-to-end, voir [Soft-NMS](/guide/soft-nms).

## Usage

```python
from ultralytics import YOLO

model = YOLO("yolo26-sea.yaml")     # scale nano par défaut
# ou
model = YOLO("yolo26s-sea.yaml")    # scale small
```
