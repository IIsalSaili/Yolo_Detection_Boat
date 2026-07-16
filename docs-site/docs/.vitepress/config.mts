import { defineConfig } from "vitepress";

export default defineConfig({
  title: "YOLO26-SEA",
  description: "Documentation du projet de détection de navires à longue distance (YOLO26-SEA)",
  lang: "fr-FR",
  base: "/",
  themeConfig: {
    nav: [
      { text: "Accueil", link: "/" },
      { text: "Architecture", link: "/guide/architecture" },
      { text: "Changements", link: "/guide/changes-overview" },
    ],
    sidebar: [
      {
        text: "Introduction",
        items: [
          { text: "Vue d'ensemble", link: "/" },
          { text: "Résumé des changements", link: "/guide/changes-overview" },
        ],
      },
      {
        text: "Architecture du modèle",
        items: [
          { text: "Architecture globale", link: "/guide/architecture" },
          { text: "SESA (backbone)", link: "/guide/sesa" },
          { text: "BiFPN (neck)", link: "/guide/bifpn" },
          { text: "Soft-NMS (post-traitement)", link: "/guide/soft-nms" },
        ],
      },
      {
        text: "Données & tâche",
        items: [
          { text: "Dataset & taxonomie OTAN", link: "/guide/dataset" },
          { text: "Estimation de direction (proposition)", link: "/guide/direction" },
        ],
      },
      {
        text: "Résultats",
        items: [{ text: "Ablation & benchmarks", link: "/guide/results" }],
      },
    ],
    socialLinks: [{ icon: "github", link: "https://github.com/ultralytics/ultralytics" }],
    search: { provider: "local" },
  },
});
