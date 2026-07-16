# Estimation de direction (proposition, non implémentée)

::: warning Statut
Ceci est une **proposition d'annotation et d'entraînement**, documentée dans le cahier des charges du
projet. Aucun code d'entraînement / branche de tête dédiée à la direction n'est présent dans le dépôt
à ce jour.
:::

## Objectif

Estimer la direction de déplacement des navires détectés, en plus de la classe et de la boîte.

## Formats d'annotation envisagés

- **Option A — angle (degrés)** : étendre chaque ligne YOLO avec `angle_deg` (0–360).
  ```
  0 0.512 0.423 0.045 0.030 135.0
  ```
- **Option B — vecteur unitaire** : étendre la ligne YOLO avec `vx vy` (normalisés), pour éviter la
  discontinuité angulaire à 359°→0°.
  ```
  0 0.512 0.423 0.045 0.030 0.7071 -0.7071
  ```
- **Option C — OBB** : stocker `cx cy w h angle` (boîtes orientées).

## Recommandation d'entraînement

Prédire la direction comme un vecteur `(cos θ, sin θ)` et utiliser une perte L2 / MAE sur ces
composantes plutôt que de régresser l'angle brut — cela évite les sauts à la transition 359°→0°.

**Métrique de suivi** : erreur angulaire moyenne (MAE direction), calculée à partir de l'angle
reconstruit depuis `(cos θ, sin θ)`.

## Robustesse — augmentations synthétiques recommandées

**Pluie** :
```python
A(x) = 0.6 * N(x) + 0.4 * M(x)   # angles : -45°, 0°, 45°
```

**Brouillard** (modèle de diffusion atmosphérique) :
```python
I(x) = J(x) * t(x) + A(x) * (1 - t(x))
t(x) = exp(-β * d(x))              # β ∈ {0.03, 0.05, 0.07}
```

**Faible luminosité** (gamma transform) :
```python
F(x) = x ** γ                      # γ ∈ {1.5, 2.0, 3.0}
```
