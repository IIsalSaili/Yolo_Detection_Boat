# Résumé des changements vs YOLO26 de base

Ce projet part du modèle **YOLO26** (Ultralytics, base end-to-end NMS-free) et lui applique les
modifications décrites par le papier **YOLO-SEA** (*Entropy* 2025, 27, 667 — Deng et al.), à l'origine
pensées pour YOLOv8n, portées ici sur l'architecture YOLO26.

| # | Changement | Fichier(s) | Statut |
|---|-----------|-----------|--------|
| 1 | Bloc d'attention **SESA** (SENetV2 + SimAM) inséré dans le backbone avant `SPPF` | `ultralytics/nn/modules/attention.py` | ✅ Implémenté |
| 2 | Neck **BiFPN** à 4 niveaux (P2-P5) avec fusion pondérée apprise, remplace le PAN | `ultralytics/nn/modules/block.py` (`BiFPNAdd2`, `BiFPNAdd3`) | ✅ Implémenté |
| 3 | Config modèle dédiée référençant SESA + BiFPN | `ultralytics/cfg/models/26/yolo26-sea.yaml` | ✅ Implémenté |
| 4 | **Soft-NMS** (decay gaussien/linéaire) en alternative au NMS dur | `ultralytics/utils/nms.py` | ✅ Implémenté (désactivé par défaut, `USE_SOFT_NMS`) |
| 5 | Estimation de la **direction de déplacement** des navires | — | 📝 Proposition (annotation + entraînement), non codée |
| 6 | Taxonomie de classification 10 types OTAN pour navires militaires | — | 📝 Spécification de dataset, à appliquer lors de l'annotation |

## Différences avec le papier d'origine

Le papier YOLO-SEA modifie **YOLOv8n**. Ce projet porte les mêmes idées sur **YOLO26**, qui est déjà
end-to-end (NMS-free, `reg_max=1`, pas de DFL) :

- Le neck couvre **P2 à P5** (pas P3 à P7) : les niveaux P6/P7 (grandes cibles proches) sont retirés
  car hors de portée pour de la surveillance maritime longue distance ; un niveau **P2** est ajouté pour
  capter les navires les plus petits/lointains, conformément à la recommandation "petites cibles" du
  cahier des charges.
- Le **Soft-NMS** n'est utile que pour un modèle non end-to-end : sur YOLO26-SEA (`end2end: True`),
  il reste implémenté et disponible via un flag global, mais n'est pas actif par défaut puisque le
  modèle NMS-free ne repasse pas par cette étape en inférence standard.
- **C2PSA** (attention spatiale native de YOLO26) est conservé après le `SPPF`, en complément de SESA.

Voir [Architecture globale](/guide/architecture) pour le détail couche par couche, ou l'onglet
correspondant à chaque modification pour les formules et le code.
