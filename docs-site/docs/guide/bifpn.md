# BiFPN — neck pondéré bidirectionnel

Fichier : `ultralytics/nn/modules/block.py` (`BiFPNAdd2`, `BiFPNAdd3`)

Le neck remplace le PAN standard par un **BiFPN à 4 niveaux (P2-P5)**, avec fusion pondérée apprenante
(*fast normalized fusion*, ε = 1e-4) :

```
p_i^td  = Conv((w1·p_i^in + w2·Resize(p_{i+1}^in)) / (w1 + w2 + ε))
p_i^out = Conv((w'1·p_i^in + w'2·p_i^td + w'3·Resize(p_{i-1}^out)) / (w'1+w'2+w'3+ε))
```

## Modules

- **`BiFPNAdd2`** — fusion de 2 features (utilisé sur les nœuds top-down et le nœud de sortie P5,
  qui n'a pas de skip d'entrée intermédiaire) :

```python
class BiFPNAdd2(nn.Module):
    def __init__(self, eps=1e-4):
        self.w = nn.Parameter(torch.ones(2), requires_grad=True)
    def forward(self, x):
        w = torch.relu(self.w)
        w = w / (w.sum() + self.eps)
        return w[0] * x[0] + w[1] * x[1]
```

- **`BiFPNAdd3`** — fusion de 3 features (nœuds de sortie intermédiaires P3/P4 : input + top-down +
  bottom-up = la connexion "skip" caractéristique de BiFPN) :

```python
class BiFPNAdd3(nn.Module):
    def forward(self, x):
        w = torch.relu(self.w)
        w = w / (w.sum() + self.eps)
        return w[0] * x[0] + w[1] * x[1] + w[2] * x[2]
```

## Câblage dans `yolo26-sea.yaml`

| Niveau | Nœud | Fusion | Cible |
|--------|------|--------|-------|
| P5 → P4 | `P4_td` | `BiFPNAdd2([P4_in, up(P5_in)])` | — |
| P4 → P3 | `P3_td` | `BiFPNAdd2([P3_in, up(P4_td)])` | — |
| P3 → P2 | `P2_out` | `BiFPNAdd2([P2_in, up(P3_td)])` | navires très petits / lointains |
| P2 → P3 | `P3_out` | `BiFPNAdd3([P3_in, P3_td, down(P2_out)])` | navires petits |
| P3 → P4 | `P4_out` | `BiFPNAdd3([P4_in, P4_td, down(P3_out)])` | navires moyens |
| P4 → P5 | `P5_out` | `BiFPNAdd2([P5_in, down(P4_out)])` | navires grands / proches |

Contrairement au papier d'origine (P3-P7, 5 niveaux, pensé pour YOLOv8n), ce projet utilise **P2-P5**
(4 niveaux) : le niveau P2 supplémentaire (stride 4, résolution la plus fine) cible spécifiquement les
navires lointains à faible emprise pixel, et les niveaux P6/P7 (grandes cibles proches) sont retirés car
hors du cas d'usage (surveillance longue distance).

Toutes les branches sont projetées à **256 canaux** avant fusion (convolutions 1×1, nœuds 12-15 du
YAML) pour uniformiser les tailles de tenseurs.

::: warning Point de vigilance
`torch.relu(w) / (w.sum() + eps)` peut produire une division proche de zéro si tous les poids
convergent vers 0 pendant l'entraînement (risque de NaN). À surveiller sur les runs longs.
:::
