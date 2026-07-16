# SESA — attention backbone

Fichier : `ultralytics/nn/modules/attention.py`

SESA fusionne deux mécanismes d'attention **channel-preserving** (`c_out == c_in`), insérés dans le
backbone après le dernier bloc `C3k2` (index 8) et avant `SPPF` (index 10) :

```yaml
- [-1, 2, C3k2, [1024, True]]        # 8
- [-1, 1, SESA, []]                  # 9   SESA attention
- [-1, 1, SPPF, [1024, 5, 3, True]]  # 10
```

## 1. SENetV2 — attention canaux

Squeeze-and-excitation "agrégée" : plusieurs branches parallèles (`cardinality`) compressent le
vecteur de canaux (global average pooling), sont concaténées puis ré-excitées via un gate sigmoïde.

```
SENetV2(x) = x · σ(W_excite · concat(branch_1(GAP(x)), …, branch_c(GAP(x))))
```

```python
class SENetV2(nn.Module):
    def __init__(self, c1, reduction=16, cardinality=4):
        ...
    def forward(self, x):
        y = self.avgpool(x).view(b, c)
        y = torch.cat([branch(y) for branch in self.branches], dim=1)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
```

## 2. SimAM — attention spatiale, zéro paramètre

Calcule une énergie par neurone à partir de l'auto-similarité locale dans chaque canal, puis pondère
l'activation par l'inverse de cette énergie (sigmoïde). **N'ajoute aucun paramètre appris.**

```
e*_t = 4(σ² + λ) / ((t − μ̂)² + 2σ² + 2λ)
X̃    = sigmoid(1/E) · X
```

```python
class SimAM(nn.Module):
    def forward(self, x):
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activation(y)
```

## 3. Fusion SESA

```python
class SESA(nn.Module):
    def __init__(self, c1, reduction=16, cardinality=4, e_lambda=1e-4):
        self.se = SENetV2(c1, reduction, cardinality)
        self.simam = SimAM(c1, e_lambda)

    def forward(self, x):
        return self.simam(self.se(x))   # canaux d'abord, spatial ensuite
```

L'ordre est important : SENetV2 (canaux) est appliqué **en premier**, SimAM (spatial) **ensuite** —
c'est l'ordre validé par l'ablation du papier YOLO-SEA (backbone-only : mAP@0.5 = 86,3%).

::: warning Point de vigilance (production / INT8)
D'après l'analyse d'architecture interne du projet, `SimAM` utilise des opérations (moyenne/variance
par canal, division) qui peuvent être problématiques à quantifier en INT8 sur certaines cibles embarquées
(ex. Jetson). À vérifier avant tout export INT8.
:::
