# Guide annotateur — Projet YOLO-SEA Maritime

## Contexte

Tu vas annoter des images de navires filmés depuis le niveau de la mer, dans des conditions variées (grand large, brume, contre-jour, nuit). L'objectif est d'entraîner un modèle de détection et de classification de navires à longue distance.

Certaines images montrent des navires très lointains (quelques pixels) — c'est normal, annote quand même.

---

## Interface Label Studio

L'interface est organisée en **4 zones** :

```
[    IMAGE    ] [ Détection  ] [ Nationalité ] [ Keypoints  ]
               [ Statut     ]               [ Conditions ]
               [ Type       ]
[ BARRE RACCOURCIS — une ligne en bas ]
```

Les zones de droite ne s'affichent que quand une boîte est sélectionnée (sauf Conditions qui s'applique à l'image entière).

---

## Flux de travail — dans cet ordre

### ① Détecter `[S]`

Appuie sur `S`, trace une boîte autour de **chaque navire visible** — même flou, même tout petit, même coupé par le bord.

Ne pas annoter : rochers, vagues, bouées fixes, phares, quais.

---

### ② Statut civil / militaire

Sélectionne la boîte → panneau central → une touche :

| Touche | Valeur | Quand |
|--------|--------|-------|
| `C` | civil | Cargo, ferry, voilier, pêcheur... |
| `M` | militaire | Frégate, destroyer, patrouilleur... |
| `U` | inconnu | Trop loin, trop flou, impossible à trancher |

**En cas de doute → `U`. Ne jamais deviner.**

---

### ③ Type de navire

Sélectionne le type **uniquement si tu peux le déterminer avec certitude**. Laisse vide si trop loin ou trop flou — c'est normal.

**10 types militaires OTAN :**

| Touche | Code OTAN | Type |
|--------|-----------|------|
| `A` | CV / CVN | Porte-avions |
| `H` | CG | Croiseur |
| `D` | DD / DDG | Destroyer |
| `F` | FF / FFG | Frégate |
| `V` | FS | Corvette |
| `P` | PC / PG | Patrouilleur |
| `O` | SS / SSN / SSBN | Sous-marin |
| `Y` | LHA / LHD / LPD | Bâtiment amphibie |
| `J` | AO / AOR | Bâtiment de soutien |
| `X` | MCM | Guerre des mines |

**Civil :**

| Touche | Type |
|--------|------|
| `G` | civil *(classe unique — la classification fine viendra plus tard)* |

---

### ④ Nationalité (si identifiable)

Indique la nationalité uniquement si elle est clairement identifiable — pavillon lisible, marquages visibles, silhouette très caractéristique. **Laisse vide sinon.**

| Touche | Nationalité |
|--------|------------|
| `W` | France |
| `Z` | USA |
| `K` | Royaume-Uni |
| `L` | Russie |
| `N` | Chine |
| — | Autres (clic souris) |

---

### ⑤ Keypoints (si navire > 50px de large)

Place les points sur les parties visibles du navire **uniquement si le navire est assez grand**. Un point manquant vaut mieux qu'un point mal placé.

| Touche | Point | Description |
|--------|-------|-------------|
| `1` | Proue | Avant du navire |
| `2` | Poupe | Arrière du navire |
| `3` | Mât | Point le plus haut |
| `4` | Passerelle | Poste de commandement |
| `5` | Babord | Flanc gauche (vu de l'arrière) |
| `6` | Tribord | Flanc droit |

Le vecteur proue → poupe donne le cap du navire.

---

### ⑥ Conditions de l'image (une fois par image)

Remplis **une seule fois par image**, pour le navire principal.

**Météo :**

| Touche | Condition |
|--------|-----------|
| `F1` | Dégagé |
| `F2` | Brume |
| `F3` | Brouillard |
| `F4` | Pluie |
| `F5` | Nuit |
| `F6` | Contre-jour |

**Distance estimée :**

| Touche | Distance |
|--------|---------|
| `7` | Proche |
| `8` | Loin |
| `9` | Très loin |

---

### ⑦ Valider `[Entrée]`

Appuie sur `Entrée` pour valider et passer à l'image suivante.

---

## Récapitulatif raccourcis clavier

| Action | Touche |
|--------|--------|
| Dessiner une boîte | `S` + glisser |
| Désélectionner | `Échap` |
| Supprimer région sélectionnée | `Suppr` |
| Valider + image suivante | `Entrée` |
| **Statut** | `C` civil · `M` militaire · `U` inconnu |
| **Type militaire** | `A`CV · `H`CG · `D`DD · `F`FF · `V`FS · `P`PC · `O`SS · `Y`LHA · `J`AO · `X`MCM |
| **Type civil** | `G` civil |
| **Nationalité** | `W`FR · `Z`US · `K`UK · `L`RU · `N`CN |
| **Keypoints** | `1`proue · `2`poupe · `3`mât · `4`passerelle · `5`babord · `6`tribord |
| **Météo** | `F1`dégagé · `F2`brume · `F3`brouillard · `F4`pluie · `F5`nuit · `F6`contre-jour |
| **Distance** | `7`proche · `8`loin · `9`très loin |

---

## Règles essentielles

1. **Annoter TOUS les navires visibles** — même flous, partiels, ou minuscules.
2. **Ne jamais annoter autre chose** qu'un navire.
3. **Laisser vide ce qu'on ne peut pas déterminer** — ne pas inventer.
4. **Keypoints seulement si navire > ~50px de large.**
5. **Doute sur le statut → `U` Inconnu.**
6. **Doute sur le type ou la nationalité → laisser vide.**

---

## Exemple d'annotation

**Image : frégate française au premier plan + petit navire lointain en arrière-plan**

- Navire 1 (grand) : `S` → boîte → `M` → `F` (frégate) → `W` (France) → keypoints `1` à `6`
- Navire 2 (lointain) : `S` → boîte → `U` → rien d'autre
- Conditions image : `F1` (dégagé) → `8` (loin)
- `Entrée` pour valider

Durée estimée : **15–45 secondes par image** selon le nombre de navires.

---

*Pour tout cas inhabituel (pennant lisible, navire ambigu, conditions extrêmes) : noter dans le champ de recherche Label Studio ou signaler à l'équipe.*
