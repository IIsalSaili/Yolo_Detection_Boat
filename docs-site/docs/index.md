---
layout: home
hero:
  name: YOLO26-SEA
  text: Détection de navires à longue distance
  tagline: Backbone YOLO26 end-to-end + attention SESA + neck BiFPN + Soft-NMS, spécialisés pour les petites cibles maritimes.
  actions:
    - theme: brand
      text: Vue d'ensemble des changements
      link: /guide/changes-overview
    - theme: alt
      text: Architecture détaillée
      link: /guide/architecture
features:
  - title: SESA (backbone)
    details: SENetV2 (canaux) + SimAM (spatial, 0 paramètre) insérés avant SPPF pour renforcer la sensibilité aux petites cibles.
  - title: BiFPN P2-P5
    details: Fusion pondérée apprenante bidirectionnelle, avec un niveau P2 supplémentaire dédié aux navires lointains.
  - title: Soft-NMS
    details: Décroissance de confiance (gaussienne ou linéaire) au lieu d'une suppression dure, activable sans réentraînement.
---
