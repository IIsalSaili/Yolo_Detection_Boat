# CLAUDE.md — Project : détection de navires à haute distance

## Contexte du project

Implémentation d'un modèle de détection de navires à **longue distance** basé sur **YOLOv8** (repo Ultralytics), en s'inspirant des modifications décrites dans le papier **YOLO-SEA** (_Entropy_ 2025, 27, 667 — Deng et al., Shanghai Maritime University / CAS).

L'objectif est d'améliorer la détection de **petites cibles maritimes** (navires lointains, faible résolution apparente) et d'estimer leur **direction / orientation de déplacement**, dans des conditions potentiellement adverses (brouillard, contre-jour, faible luminosité).

---

## Référence technique principale

**Papier :** YOLO-SEA — An Enhanced Detection Framework for Multi-Scale Maritime Targets in Complex Sea States and Adverse Weather  
**DOI :** https://doi.org/10.3390/e27070667  
**Base :** YOLOv8n (version nano — 3,01M params, 8,2 GFLOPs)  
**Résultats publiés :** mAP@0.5 = 88,2% (+5,8 pt vs baseline), mAP@0.5:0.95 = 60,8% (+7,2 pt), params réduits à 2,80M

---

## Les 3 modifications à implémenter (par ordre de priorité)

### 1. Module SESA — backbone

Insérer après le dernier bloc `C2f` du backbone, avant `SPPF`.

**SENetV2** (`modules/attention.py`) — recalibration des canaux :

- Global average pooling → compression FC (ratio r) → branches FC parallèles → gating léger
- Formule : `SENetV2 = x + F(x · Ex(Σ Sq(x)))`

**SimAM** (`modules/attention.py`) — attention spatiale sans paramètres :

- Function d'énergie par neurone : `e*_t = 4(σ² + λ) / ((t − μ̂)² + 2σ² + 2λ)`
- Application : `X̃ = Sigmoid(1/E) · X`
- Zéro paramètre supplémentaire — ne pas changer le nombre total de params du backbone

**Fusion SESA** = SENetV2 appliqué en premier (canaux), SimAM ensuite (spatial). Enregistrer dans `ultralytics/nn/modules/` et déclarer dans `__init__.py`.

Position testée optimale (ablation du papier) : **backbone uniquement** → mAP@0.5 = 86,3% (vs 82,8% baseline avec juste SENetV2+BiFPN sans SimAM).

### 2. BiFPN amélioré — neck

Remplacer le `PAN` du neck par un `BiFPN` à **5 niveaux** :

| Niveau | Cible           | Nb couches |
| ------ | --------------- | ---------- |
| P3     | petites cibles  | ×2         |
| P4     | petites cibles  | ×2         |
| P5     | cibles moyennes | ×2         |
| P6     | cibles moyennes | ×1         |
| P7     | grandes cibles  | ×1         |

Formulas du fast normalized fusion (ε = 0,0001) :

```
p_i^td  = Conv((w1·p_i^in + w2·Resize(p_{i+1}^in)) / (w1 + w2 + ε))
p_i^out = Conv((w'1·p_i^in + w'2·p_i^td + w'3·Resize(p_{i-1}^out)) / (w'1+w'2+w'3+ε))
```

Les poids `w` sont apprenants. Connexions bidirectionnelles (top-down ET bottom-up) avec sauts entre input et output du même niveau.

Implémenter dans `ultralytics/nn/modules/block.py` ou un fichier `bifpn.py` dédié. Modifier le fichier `.yaml` de configuration du modèle pour référencer le nouveau neck.

### 3. Soft-NMS — post-traitement

Remplacer le NMS classique dans `ultralytics/utils/ops.py` (function `non_max_suppression`).

Formule de decay avec reset gaussien :

```python
# Classique NMS : suppression dure si IoU >= Nt
# Soft-NMS : atténuation de confiance
if iou(M, b_i) >= Nt:
    s_i = s_i * (1 - iou(M, b_i))  # linear decay
    # ou gaussien : s_i = s_i * exp(-iou²/sigma)
```

**Important :** Soft-NMS ne nécessite **aucun réentraînement** — peut être testé sur un checkpoint existent. Tester en dernier pour avoir une measure de gain propre.

---

## Architecture YOLO-SEA résumée

```
Input 640×640
    │
    ▼
[BACKBONE]
  Conv → C2f×1 → C2f×2 → C2f×2 → SPPF
                                      │
                                   [SESA]  ← insertion ici
                                      │
    ▼
[NECK — BiFPN 5 niveaux]
  P3 (×2) ──┐
  P4 (×2)   ├── bidirectionnel + weighted fusion
  P5 (×2)   │
  P6 (×1)   │
  P7 (×1) ──┘
    │
    ▼
[HEAD — Decoupled]
  Task-aligned assigner
  Box loss (CIoU + DFL) + CLS loss (BCE)
  Direction estimation branch (régression de l'orientation / cap)
    │
    ▼
[POST-TRAITEMENT]
  Threshold filter → Soft-NMS gaussien
    │
    ▼
  Détections finales
```

---

## Dataset

### Dataset de référence (papier)

- **Source :** Roboflow
- **Volume :** 1 934 images
- **6 classes :** BRF, CDG, FDA, FLF, FREMM, PHA
- **Split :** 70% train / 20% val / 10% test
- **Annotation :** LabelMe → format YOLO (txt)

### Pour ce project

Le project se concentre sur la **détection de navires à longue distance**. Les cibles apparaissent petites dans l'image (zone de pixels réduite).

### Taxonomie de classification — 10 types OTAN (militaires)

La tête de classification de type utilize la nomenclature standardisée OTAN :

| Code OTAN             | Type                | Description                                |
| --------------------- | ------------------- | ------------------------------------------ |
| CV / CVN              | Porte-avions        | Conventionnel / nucléaire                  |
| CG / CGN              | Croiseur            | Escorteur lourd à missiles                 |
| DD / DDG              | Destroyer           | Escorteur polyvalent                       |
| FF / FFG              | Frégate             | Escorteur anti-sous-marin / ASM            |
| FS / FSG              | Corvette            | Bâtiment de combat côtier                  |
| PC / PG / PTG         | Patrouilleur        | Vedette côtière / garde-côtes              |
| SS / SSN / SSBN       | Sous-marin          | Conventionnel / nucléaire d'attaque / SNLE |
| LHA / LHD / LPD / LST | Bâtiment amphibie   | Projection de force / débarquement         |
| AO / AOR / AE / AS    | Bâtiment de soutien | Ravitaillement / réparation                |
| MCM / MHC / MSO       | Guerre des mines    | Dragueur / chasseur de mines               |

Ces 10 catégories constituent la **classe `ship_type` pour les navires militaires**. Les navires civils restent classés par type fonctionnel (cargo, pétrolier, porte-conteneurs, etc.) sans code OTAN.

### Annotation directionnelle (proposition)

Pour permettre l'estimation de la direction de déplacement, ajouter un vecteur directionnel ou un angle aux annotations. Options possibles :

- Option A — angle (degrés) : étendre chaque ligne YOLO (`class x_center y_center width height`) avec `angle_deg` (0–360).
- Option B — vecteur unitaire : étendre la ligne YOLO avec `vx vy` (normalisés), utile pour éviter la discontinuité angulaire.
- Option C — OBB : stocker `cx cy w h angle` si vous préférez les boîtes orientées.

Examples (format texte YOLO proposé) :

```
# Option A (angle en degrés)
0 0.512 0.423 0.045 0.030 135.0

# Option B (vecteur unitaire)
0 0.512 0.423 0.045 0.030 0.7071 -0.7071
```

Recommendation d'entraînement : prédire la direction comme vecteur `(cos θ, sin θ)` ou comme vecteur unitaire `(vx, vy)` et utiliser une perte L2 / MAE sur ces composantes — cela évite les sauts à la transition 359°→0°. Measure à suivre : erreur angulaire moyenne (MAE direction) calculée à partir de l'angle reconstruit.

**Augmentations synthétiques recommandées pour la robustesse :**

Pluie :

```python
A(x) = 0.6 * N(x) + 0.4 * M(x)   # angles : -45°, 0°, 45°
```

Brouillard (modèle diffusion atmosphérique) :

```python
I(x) = J(x)*t(x) + A(x)*(1-t(x))
t(x) = exp(-β * d(x))              # β ∈ {0.03, 0.05, 0.07}
```

Faible luminosité (gamma transform) :

```python
F(x) = x ** γ                      # γ ∈ {1.5, 2.0, 3.0}
```

---

## Résultats de référence (ablation complète)

| Config               | SESA  | BiFPN | Soft-NMS | mAP@0.5   | mAP@0.5:0.95 | Params (M) | FLOPs (G) |
| -------------------- | ----- | ----- | -------- | --------- | ------------ | ---------- | --------- |
| Baseline YOLOv8n     | —     | —     | —        | 82,4%     | 53,6%        | 3,01       | 8,2       |
| + SESA seul          | ✓     | —     | —        | 84,2%     | 56,8%        | 3,03       | 8,2       |
| + BiFPN seul         | —     | ✓     | —        | 84,1%     | 54,9%        | 2,79       | 8,2       |
| + Soft-NMS seul      | —     | —     | ✓        | 85,0%     | 57,6%        | 3,01       | 8,9       |
| SESA + BiFPN         | ✓     | ✓     | —        | 86,3%     | 56,4%        | 2,80       | 8,1       |
| **YOLO-SEA complete** | **✓** | **✓** | **✓**    | **88,2%** | **60,8%**    | **2,80**   | **8,1**   |

**Résultats météo adverse (test synthétique pluie + brouillard + nuit) :**

| Modèle   | P     | R     | mAP@0.5 | mAP@0.5:0.95 |
| -------- | ----- | ----- | ------- | ------------ |
| YOLOv8n  | 68,0% | 73,8% | 75,4%   | 45,7%        |
| YOLO-SEA | 73,4% | 80,7% | 82,9%   | 56,9%        |

---

## Environment de travail

**Stack de référence (papier) :**

- OS : Windows 11
- Framework : PyTorch 2.4.1
- GPU : NVIDIA RTX 4060 Ti ( 8 Go VRAM)
- Entrée : 640×640

**Hyperparamètres d'entraînement (papier) :**

- Epochs : 300
- Optimizer : SGD, lr = 0,01, momentum = 0,01
- Paramètres Ultralytics : défauts YOLOv8 pour le reste

**Repo :** Ultralytics YOLOv8 — https://github.com/ultralytics/ultralytics  
**Fichiers clés à modifier :**

- `ultralytics/nn/modules/__init__.py` — déclarer SESA, BiFPN
- `ultralytics/nn/modules/block.py` — implémenter BiFPN
- `ultralytics/nn/modules/attention.py` (à créer) — SENetV2 + SimAM + SESA
- `ultralytics/utils/ops.py` — remplacer NMS par Soft-NMS
- `ultralytics/cfg/models/v8/yolov8-sea.yaml` — config du modèle custom

---

## Métriques à suivre

Métriques principales : `mAP@0.5`, `mAP@0.5:0.95`, erreur angulaire moyenne (MAE direction)  
Métriques secondaires : Précision (P), Rappel (R), F1  
Métriques modèle : nombre de paramètres (M), FLOPs (G), vitesse d'inférence (ms/img)  
Tests de robustesse : évaluer sur jeu de test avec dégradations synthétiques

---

## Plan de route

| Phase | Contenu                                             | Durée estimée |
| ----- | --------------------------------------------------- | ------------- |
| 1     | Dataset + baseline YOLOv8n                          | S1–S2         |
| 2     | Implémenter + tester SESA (backbone)                | S2–S3         |
| 3     | Implémenter + tester BiFPN (neck)                   | S3–S4         |
| 4     | Brancher Soft-NMS + ablation complète + tests météo | S5–S6         |

Ordre impératif : dataset stable avant toute modification → SESA avant BiFPN → Soft-NMS en dernier (pas de réentraînement).

---

## Points de vigilance

- **SESA :** tester SENetV2 et SimAM comme modules indépendants avant fusion. Vérifier que SimAM n'introduit aucun paramètre.
- **BiFPN :** les connexions skip entre input/output du même niveau sont la partie la plus délicate à câbler correctement dans Ultralytics. Vérifier que P3/P4 reçoivent bien les feature maps intermédiaires du backbone, pas seulement la sortie finale.
- **Soft-NMS :** peut être appliqué sans réentraînement sur n'importe quel checkpoint — tester en premier sur le baseline pour mesurer le gain en isolation.
- **yaml custom :** nommer le fichier de config `yolov8-sea.yaml` (ou similaire) pour éviter d'écraser les configs Ultralytics par défaut.
- **Petites cibles :** pour ce project (navires lointains), les niveaux P3/P4 du BiFPN sont critiques. Si les performances sur petites cibles restent insuffisantes, envisager d'ajouter une couche de détection supplémentaire à plus haute résolution.
- **Généralisation :** tester sur un second dataset public (ex. 6_class_final sur Roboflow) pour valider que le modèle ne sur-apprend pas le dataset principal.
