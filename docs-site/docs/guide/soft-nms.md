# Soft-NMS — post-traitement

Fichier : `ultralytics/utils/nms.py`

Remplace la suppression dure du NMS classique par une **décroissance de la confiance** des boîtes
chevauchantes, sans supprimer directement les détections :

```python
# NMS classique : suppression dure si IoU >= Nt
# Soft-NMS       : atténuation de confiance
if iou(M, b_i) >= Nt:
    s_i = s_i * (1 - iou(M, b_i))            # decay linéaire
    # ou :
    s_i = s_i * exp(-iou(M, b_i)**2 / sigma)  # decay gaussien (par défaut)
```

## Configuration

```python
# ultralytics/utils/nms.py
USE_SOFT_NMS = False           # activer Soft-NMS globalement
SOFT_NMS_SIGMA = 0.5           # sigma du decay gaussien
SOFT_NMS_METHOD = "gaussian"   # "gaussian" ou "linear"
```

L'implémentation vit dans `TorchNMS.soft_nms(boxes, scores, iou_thres, sigma=..., method=..., score_thres=...)`
et s'active via les paramètres `soft` / `soft_sigma` de la fonction principale de NMS.

## Pourquoi c'est optionnel ici

`yolo26-sea.yaml` déclare `end2end: True` — le modèle YOLO26 est **NMS-free** en inférence standard
(assignation one-to-one côté entraînement). Soft-NMS ne s'applique donc que si le modèle est exporté /
utilisé en mode **non end-to-end**. Il reste implémenté et activable via `USE_SOFT_NMS` pour ce cas,
et pour comparer les gains en isolation sur un baseline (ex. YOLOv8n) sans réentraînement.

::: tip
Soft-NMS ne nécessite **aucun réentraînement** — il peut être testé directement sur un checkpoint déjà
entraîné, ce qui permet de mesurer son gain de façon isolée avant de l'ajouter à une ablation complète.
:::
