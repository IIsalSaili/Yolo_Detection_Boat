# Dataset & taxonomie OTAN

## Dataset de référence (papier YOLO-SEA)

- Source : Roboflow
- Volume : 1 934 images
- 6 classes : `BRF`, `CDG`, `FDA`, `FLF`, `FREMM`, `PHA`
- Split : 70% train / 20% val / 10% test
- Annotation : LabelMe → format YOLO (`.txt`)

## Ce projet

Se concentre sur la **détection de navires à longue distance** : les cibles occupent une zone de
pixels réduite dans l'image, d'où l'importance du niveau P2 du BiFPN et de l'attention SESA.

## Taxonomie de classification — 10 types OTAN (navires militaires)

| Code OTAN | Type | Description |
|-----------|------|--------------|
| CV / CVN | Porte-avions | Conventionnel / nucléaire |
| CG / CGN | Croiseur | Escorteur lourd à missiles |
| DD / DDG | Destroyer | Escorteur polyvalent |
| FF / FFG | Frégate | Escorteur anti-sous-marin / ASM |
| FS / FSG | Corvette | Bâtiment de combat côtier |
| PC / PG / PTG | Patrouilleur | Vedette côtière / garde-côtes |
| SS / SSN / SSBN | Sous-marin | Conventionnel / nucléaire d'attaque / SNLE |
| LHA / LHD / LPD / LST | Bâtiment amphibie | Projection de force / débarquement |
| AO / AOR / AE / AS | Bâtiment de soutien | Ravitaillement / réparation |
| MCM / MHC / MSO | Guerre des mines | Dragueur / chasseur de mines |

Ces 10 catégories constituent la classe `ship_type` pour les navires militaires. Les navires civils
restent classés par type fonctionnel (cargo, pétrolier, porte-conteneurs, etc.) sans code OTAN.

::: info Statut
Cette taxonomie est une **spécification d'annotation**, à appliquer lors de la constitution / extension
du dataset. Aucune tête de classification `ship_type` dédiée n'est encore branchée dans le modèle.
:::
