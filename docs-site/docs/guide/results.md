# Ablation & benchmarks (papier de référence)

Résultats publiés par le papier YOLO-SEA sur YOLOv8n (référence pour ce projet ; les chiffres
spécifiques à YOLO26-SEA restent à mesurer sur ce dépôt) :

| Config | SESA | BiFPN | Soft-NMS | mAP@0.5 | mAP@0.5:0.95 | Params (M) | FLOPs (G) |
|--------|:---:|:---:|:---:|---:|---:|---:|---:|
| Baseline YOLOv8n | — | — | — | 82,4% | 53,6% | 3,01 | 8,2 |
| + SESA seul | ✓ | — | — | 84,2% | 56,8% | 3,03 | 8,2 |
| + BiFPN seul | — | ✓ | — | 84,1% | 54,9% | 2,79 | 8,2 |
| + Soft-NMS seul | — | — | ✓ | 85,0% | 57,6% | 3,01 | 8,9 |
| SESA + BiFPN | ✓ | ✓ | — | 86,3% | 56,4% | 2,80 | 8,1 |
| **YOLO-SEA complet** | **✓** | **✓** | **✓** | **88,2%** | **60,8%** | **2,80** | **8,1** |

## Robustesse météo adverse (test synthétique pluie + brouillard + nuit)

| Modèle | P | R | mAP@0.5 | mAP@0.5:0.95 |
|--------|---:|---:|---:|---:|
| YOLOv8n | 68,0% | 73,8% | 75,4% | 45,7% |
| YOLO-SEA | 73,4% | 80,7% | 82,9% | 56,9% |

## Métriques suivies dans ce projet

- **Principales** : `mAP@0.5`, `mAP@0.5:0.95`, erreur angulaire moyenne (MAE direction, une fois implémentée)
- **Secondaires** : Précision (P), Rappel (R), F1
- **Modèle** : nombre de paramètres (M), FLOPs (G), vitesse d'inférence (ms/img)
- **Robustesse** : évaluation sur jeu de test avec dégradations synthétiques (pluie, brouillard, faible luminosité)
