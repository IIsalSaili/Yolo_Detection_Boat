# Recherche d'architecture IA — Détection et classification de navires à longue distance

> Synthèse appuyée sur l'état de l'art 2024–2026, reliée à la base existante du project
> (YOLO-SEA sur Ultralytics, contrainte nano/small pour l'embarqué INT8).

## TL;DR — recommendation

Garder un détecteur dense one-stage type YOLO comme socle, mais migrer vers un design **end-to-end NMS-free** (style YOLO26 ou DEIM), ajouter une **tête de classification hiérarchique à probabilités conditionnelles avec abstention** (le modèle "remonte" dans la taxonomie quand la résolution est insuffisante), une **tête keypoints conditionnelle** (proue/poupe → cap géométrique), et exploiter deux priors gratuits du domaine que presque personne n'utilise : **la ligne d'horizon** (géométrie monoculaire → distance estimée → gating des têtes) et **le temporel** (caméra fixe sur la mer → accumulation multi-frames pour les cibles sub-10 px, qui donne en bonus la direction de déplacement).

---

## 1. Socle architectural : end-to-end NMS-free plutôt que Soft-NMS

Le plan actuel (YOLO-SEA) prévoit Soft-NMS en post-traitement. C'est valable pour la recherche, mais c'est à contre-courant pour l'embarqué : le NMS est la partie la plus pénible à exporter (shapes dynamiques, exécution CPU, latence variable).

- **[YOLO26](https://docs.ultralytics.com/models/yolo26)** (Ultralytics, oct. 2025 — [analyze arXiv](https://arxiv.org/abs/2601.12882)) correspond exactement au cahier des charges : NMS-free natif (−43 % de latence CPU sur le nano), **suppression du DFL** précisément pour faciliter l'export ONNX/TensorRT et la quantification, et surtout **ProgLoss + STAL** (Small-Target-Aware Label assignment) qui ciblent les petits objects. Le project étant déjà sur le repo Ultralytics, la migration est peu coûteuse et les modules SESA/BiFPN se greffent dessus de la même façon.
- Alternative transformer : **DEIM** (CVPR 2025) et **[DEIMv2](https://arxiv.org/html/2509.20787v2)** (sept. 2025), qui décline [D-FINE](https://playground.roboflow.com/models/ustc/d-fine) en variantes ultra-légères (Pico/Femto/Atto) pour l'edge, avec matching one-to-one donc pas de NMS du tout. Intéressant à benchmarker, mais l'attention des DETR quantifie moins bien en INT8 qu'un YOLO full-conv — à garder en piste B.

Point pratique : Soft-NMS reste utile comme **outil d'évaluation sans réentraînement** sur les checkpoints actuels (comme prévu dans le plan), mais ne pas le viser comme solution embarquée finale.

## 2. Petits objects à très longue distance

Par ordre de rapport gain/coût :

1. **Tête P2 (stride 4)** — une couche de détection à plus haute résolution. C'est la modification la plus rentable pour des cibles < 16 px ; déjà notée dans les points de vigilance du project. Coût en FLOPs réel mais maîtrisable en nano. Symétriquement, le P6/P7 du BiFPN sert peu pour des navires lointains — envisager un neck **P2–P5** plutôt que P3–P7 : même budget, mieux dépensé.
2. **Remplacer l'IoU par la [Normalized Gaussian Wasserstein Distance](https://arxiv.org/abs/2110.13389)** (NWD) dans l'assignation et la loss, ou utiliser **RFLA** (assignation par champ réceptif gaussien). L'IoU est quasi binaire pour une boîte de 8×5 px : un décalage d'un pixel fait tout basculer. C'est probablement _la_ raison pour laquelle YOLO de base échoue à longue distance, plus que l'architecture elle-même. Travaux récents dans la même veine : [scale-adaptive losses pour tiny objects](https://arxiv.org/pdf/2511.09891).
3. **SPD-Conv** (space-to-depth) à la place des convolutions stride-2 du backbone : aucune perte d'information par sous-échantillonnage, paramètres quasi constants, très favorable aux cibles de quelques pixels et trivial à quantifier.
4. **Attention** : le module SESA (SENetV2 + SimAM) est cohérent avec la littérature. Vérifier juste la quantification de SimAM (variance + division par canal — certains NPU le supportent mal ; voir §5). Des alternatives comme [les embeddings heatmap pour transformers small-object](https://arxiv.org/pdf/2504.13469) existent mais sont moins matures pour l'edge.
5. **Piste réellement innovante — le temporel** : une caméra de surveillance maritime est (quasi) statique et la mer est un fond texturé mais sans structure persistante. À très longue distance (cible sub-10 px), la détection mono-frame plafonne physiquement. Empiler 3–5 frames espacées (en canaux d'entrée, ou via un module récurrent léger type ConvGRU sur P2/P3) fait émerger les cibles par leur **persistence et leur movement** — l'esprit du _track-before-detect_ radar appliqué au deep learning. Bonus décisif : le déplacement inter-frames donne **la direction de déplacement** là où l'orientation visuelle est illisible.
6. **Prior géométrique de l'horizon** : en monoculaire maritime, presque tous les navires touchent la ligne d'horizon ou sont en dessous. Une mini-branche qui régresse la ligne d'horizon permet (a) de concentrer l'attention/les anchors sur une bande étroite de l'image, (b) d'estimer la **distance** de chaque détection (hauteur caméra + position verticale vs horizon), information réinjectable dans le gating des têtes (§4). Très peu exploité dans la littérature YOLO maritime — c'est un vrai axe de contribution.

## 3. Classification hiérarchique avec contraintes logiques

La taxonomie (navire → civil/militaire → type fin, codes OTAN) est un arbre. Trois familles d'approches, par complexité croissante :

**A. Softmax conditionnels par niveau (recommandé pour l'embarqué)** — le schéma "WordTree" de YOLO9000 : un softmax par nœud parent, le score final d'une feuille = produit des probabilités conditionnelles le long du chemin (`P(frégate) = P(navire) · P(militaire|navire) · P(frégate|militaire)`). Les implications logiques sont respectées **par construction** (jamais `P(porte-avions) > P(militaire)`), c'est différentiable de bout en bout, et ce ne sont que des softmax — INT8-friendly. L'inférence donne naturellement le comportement souhaité : on descend dans l'arbre tant que la confiance conditionnelle dépasse un seuil, sinon on **s'arrête au niveau parent** ("navire militaire, type indéterminé"). C'est l'abstention hiérarchique que le cas "si la distance le permet" demande.

**B. Multi-label sigmoïde + couche de cohérence** — la tête prédit un score sigmoïde pour _chaque nœud_ de la taxonomie, et une couche contrainte garantit `P(enfant) ≤ P(parent)`. C'est le cadre **[C-HMCNN](https://arxiv.org/pdf/2010.10151)** (cohérence guarantee par un module max-constraint) et son extension aux [contraintes logiques dures générales](https://arxiv.org/abs/2103.13427). Avantage sur A : gère les taxonomies non strictes (un même navire peut porter plusieurs attributes) et s'intègre directement à la tête BCE multi-label de YOLOv8. C'est l'option la plus propre théoriquement.

**C. Prototypes / embeddings hiérarchiques** — têtes à prototypes hiérarchiques ([HiProto](https://arxiv.org/html/2604.13981v1), 2026, pensé pour conditions dégradées) ou embeddings hyperboliques où la hiérarchie est encodée dans la géométrie de l'espace latent. Élégant et performant en conditions difficiles, mais plus exploratoire et moins évident à quantifier — piste de recherche plutôt que de production.

Dans tous les cas, entraîner avec une **loss hiérarchique** : une frégate prédite "destroyer" doit coûter moins cher qu'une frégate prédite "cargo" (pondération par distance dans l'arbre, ou some des BCE à chaque niveau).

## 4. Multi-têtes conditionnelles : le gating par résolution/distance

Architecture cible proposée :

```
                          Backbone (+ SPD-Conv, + SESA)
                                      │
                          Neck BiFPN P2–P5 (fusion pondérée)
                                      │
        ┌─────────────┬───────────────┼────────────────┬──────────────┐
   Tête détection   Tête cls        Tête keypoints   Tête direction   Branche horizon
   (NMS-free,       hiérarchique    (proue, poupe,   (cos θ, sin θ)   (régression ligne)
   STAL/NWD)        conditionnelle   mât, point…)
                          ▲               ▲                ▲
                          └───── gating par taille de boîte / distance estimée ─────┘
```

- **Gating** : la taille de la boîte (et/ou la distance issue de l'horizon) détermine quelles têtes sont _supervisées et évaluées_. En pratique : on masque la loss keypoints et la loss de classification fine pour les boîtes sous un seuil de surface, et à l'inférence on n'émet ces sorties que si la boîte dépasse le seuil. Simple, robuste, et ça évite que les gradients bruités des petites cibles polluent les têtes fines. La version apprise (un scalaire de "résolvabilité" prédit par boîte) est une jolie extension recherche.
- **Keypoints → cap** : le vecteur proue→poupe donne l'orientation géométriquement, comme dans [CHPDet](https://arxiv.org/pdf/2101.11189) (boîtes orientées + point de proue pour lever l'ambiguïté de direction) et les approaches [segmentation + keypoints par vote](https://link.springer.com/chapter/10.1007/978-3-030-22808-8_39). Tête style YOLOv8-pose (OKS loss), active uniquement en proximité. À longue distance, la direction vient du temporel (§2.5) ; les deux estimations se recoupent à moyenne distance — à fusionner.
- **Direction continue** : le choix `(cos θ, sin θ)` du CLAUDE.md est le bon (pas de discontinuité 359°→0°).
- **Équilibrage des tâches** : avec 4–5 losses, pondération par incertitude (Kendall et al.) ou GradNorm plutôt que des poids fixés à la main.
- Pour annoter/valider la pose à moindre coût : la fusion **AIS + images** ([dataset 6D pose marine 2025](https://arxiv.org/html/2508.14767v1)) permet de générer automatiquement cap et distance vérité-terrain avec un flux AIS — énorme économie d'annotation directionnelle.

## 5. Contraintes INT8 / embarqué — points concrets

- **Blocks reparamétrisables** (RepConv/RepVGG) : multi-branches à l'entraînement, fusionnés en convolutions 3×3 simples à l'export. Quantification triviale, gain gratuit.
- **Attention et fusion pondérée** : la _fast normalized fusion_ du BiFPN (division par some de poids) et la division/variance de SimAM sont les deux ops à risque en INT8. Plan B à prévoir : remplacer la fusion normalisée par concat + conv 1×1 à l'export, et valider SimAM en PTQ tôt dans le project plutôt qu'à la fin.
- **Activations** : SiLU quantifie correctement sur GPU/TensorRT mais mal sur certains NPU ; si la cible est un NPU (Hailo, RKNN, etc.), prévoir une variante ReLU/HardSwish.
- **QAT obligatoire** : sur un modèle nano, la PTQ perd typiquement 1–3 pt de mAP, et davantage sur les petits objects (les activations des petites cibles sont en queue de distribution, écrasées par la quantification). La quantification-aware training récupère l'essentiel.
- **Distillation** : entraîner un teacher medium (8 Go de VRAM suffisent en distillation même si l'entraînement direct est limité au nano/small) et distiller les feature maps P2/P3 vers le student nano — c'est là que vivent les petites cibles.
- Le NMS-free (§1) est lui-même un argument INT8 : tout le graphe devient statique.

## 6. Données et benchmarks

- **[WUTDet](https://arxiv.org/pdf/2604.07759)** (2026) : 100K images de navires avec petits objects denses — exactement le benchmark de généralisation qui manque au plan actuel (bien plus grand que les 1 934 images Roboflow).
- Se comparer à **[SeaLSOD-YOLO](https://pmc.ncbi.nlm.nih.gov/articles/PMC13075222/)** (YOLOv11, petits objects maritimes) et aux [variantes YOLOv11n maritimes récentes](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12788006/) en plus de YOLO-SEA — le champ a bougé depuis le papier Entropy 2025.

## Prioritisation suggérée

1. **NWD/RFLA + tête P2** sur la base actuelle — gain probable le plus élevé par euro, sans changer d'architecture.
2. **Tête hiérarchique conditionnelle (option A)** avec abstention — répond directement au besoin civil/militaire/type.
3. **Migration YOLO26 NMS-free** + QAT — verrouille la cible embarquée.
4. **Keypoints + direction avec gating** — une fois la détection stabilisée.
5. **Recherche** : branche horizon (gating géométrique) et module temporel track-before-detect — les deux contributions réellement originales publiables.

---

## Sources

- [YOLO26 — Ultralytics Docs](https://docs.ultralytics.com/models/yolo26)
- [YOLO26: An Analysis of NMS-Free End to End Framework (arXiv)](https://arxiv.org/abs/2601.12882)
- [DEIMv2 / Real-Time Object Detection Meets DINOv3 (arXiv)](https://arxiv.org/html/2509.20787v2)
- [D-FINE — Roboflow Playground](https://playground.roboflow.com/models/ustc/d-fine)
- [A Normalized Gaussian Wasserstein Distance for Tiny Object Detection (arXiv)](https://arxiv.org/abs/2110.13389)
- [Scale-Aware Relay and Scale-Adaptive Loss for Tiny Object Detection (arXiv)](https://arxiv.org/pdf/2511.09891)
- [HMPE: HeatMap Embedding for Efficient Transformer-Based Small Object Detection (arXiv)](https://arxiv.org/pdf/2504.13469)
- [Coherent Hierarchical Multi-Label Classification Networks — C-HMCNN (arXiv)](https://arxiv.org/pdf/2010.10151)
- [Multi-Label Classification Neural Networks with Hard Logical Constraints (arXiv)](https://arxiv.org/abs/2103.13427)
- [HiProto: Hierarchical Prototype Learning for Interpretable Object Detection (arXiv)](https://arxiv.org/html/2604.13981v1)
- [CHPDet: Arbitrary-Oriented Ship Detection through Center-Head Point Extraction (arXiv)](https://arxiv.org/pdf/2101.11189)
- [Ship Segmentation and Orientation Estimation Using Keypoints Detection and Voting (Springer)](https://link.springer.com/chapter/10.1007/978-3-030-22808-8_39)
- [Fusing Monocular RGB Images with AIS Data — 6D Pose Dataset for Marine Vessels (arXiv)](https://arxiv.org/html/2508.14767v1)
- [WUTDet: A 100K-Scale Ship Detection Dataset (arXiv)](https://arxiv.org/pdf/2604.07759)
- [SeaLSOD-YOLO: Lightweight Framework for Maritime Small Object Detection (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13075222/)
- [Maritime Small Target Image Detection Based on Improved YOLOv11n (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12788006/)
- [YOLO-SEA (Entropy 2025, MDPI)](https://www.mdpi.com/1099-4300/27/7/667)
