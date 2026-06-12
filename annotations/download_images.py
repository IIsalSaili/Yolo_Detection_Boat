"""
Télécharge les images Wikipedia manquantes et génère img_cache.json.
- Scanne d'abord images/ pour détecter ce qui est déjà là (manuel ou script).
- Ne télécharge que ce qui manque encore.
- img_cache.json = mapping wiki_slug → chemin relatif local.

Usage :
    cd annotations
    python download_images.py               # normal
    python download_images.py --force       # re-télécharger même si présent
"""

import argparse
import json
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

IMG_DIR = Path(__file__).parent / "images"
CACHE_FILE = Path(__file__).parent / "img_cache.json"
CACHE_JS = Path(__file__).parent / "img_cache.js"

# ── Liste complète des slugs Wikipedia (synchronisée avec le HTML) ──────────
WIKI_SLUGS = [
    "Absalon-class_frigate",
    "Abukuma-class_destroyer_escort",
    "Ada-class_corvette",
    "Admiral_Grigorovich-class_frigate",
    "Agosta-class_submarine",
    "Akizuki-class_destroyer_(2010)",
    "Akula-class_submarine",
    "Al_Riyadh-class_frigate",
    "Almirante_Brown-class_destroyer",
    "America-class_amphibious_assault_ship",
    "Anzac-class_frigate",
    "Arihant-class_submarine",
    "Arleigh_Burke-class_destroyer",
    "Asagiri-class_destroyer",
    "Astute-class_submarine",
    "Atago-class_destroyer",
    "Bay-class_landing_ship",
    "Bora-class_corvette",
    "Borei-class_submarine",
    "Brahmaputra-class_frigate",
    "Brandenburg-class_frigate",
    "Braunschweig-class_corvette",
    "Bremen-class_frigate",
    "Buyan-class_corvette",
    "Canberra-class_landing_helicopter_dock",
    "Carlo_Bergamini-class_frigate",
    "Chinese_aircraft_carrier_Fujian",
    "Chinese_aircraft_carrier_Liaoning",
    "Chungmugong_Yi_Sun-sin-class_destroyer",
    "Collins-class_submarine",
    "Constellation-class_frigate",
    "D'Estienne_d'Orves-class_aviso",
    "Daphné-class_submarine",
    "De_Zeven_Provinciën-class_frigate",
    "Delhi-class_destroyer",
    "Delta-class_submarine",
    "Diponegoro-class_corvette",
    "Dokdo-class_amphibious_assault_ship",
    "Dolphin-class_submarine",
    "Durand_de_la_Penne-class_destroyer",
    "Endurance-class_landing_ship_tank",
    "Espora-class_corvette",
    "FREMM_multipurpose_frigate",
    "Floréal-class_frigate",
    "Flyvefisken-class_patrol_vessel",
    "Formidable-class_frigate",
    "Freedom-class_littoral_combat_ship",
    "French_aircraft_carrier_Charles_de_Gaulle",
    "Fridtjof_Nansen-class_frigate",
    "Gerald_R._Ford-class_aircraft_carrier",
    "Gotland-class_submarine",
    "Gowind-class_corvette",
    "Gwanggaeto_the_Great-class_destroyer",
    "HTMS_Chakri_Naruebet",
    "Halifax-class_frigate",
    "Hamina-class_missile_boat",
    "Hatakaze-class_destroyer",
    "Hobart-class_destroyer",
    "Horizon-class_frigate",
    "Hydra-class_frigate",
    "Hyūga-class_helicopter_destroyer",
    "INS_Vikramaditya",
    "Incheon-class_frigate",
    "Independence-class_littoral_combat_ship",
    "Istanbul-class_frigate",
    "Italian_aircraft_carrier_Cavour",
    "Italian_aircraft_carrier_Giuseppe_Garibaldi",
    "Italian_landing_helicopter_dock_Trieste",
    "Ivan_Gren-class_landing_ship",
    "Iver_Huitfeldt-class_frigate",
    "Izumo-class_helicopter_destroyer",
    "Kamorta-class_corvette",
    "Kang_Ding-class_frigate",
    "Karakurt-class_corvette",
    "Karel_Doorman-class_frigate",
    "Kashin-class_destroyer",
    "Kidd-class_destroyer",
    "Kilo-class_submarine",
    "Kirov-class_battlecruiser",
    "Knud_Rasmussen-class_patrol_vessel",
    "Kolkata-class_destroyer",
    "Kongō-class_destroyer",
    "Kora-class_corvette",
    "Krivak-class_frigate",
    "Kuznetsov-class_aircraft_carrier",
    "La_Fayette-class_frigate",
    "Lada-class_submarine",
    "Lekiu-class_frigate",
    "Los_Angeles-class_submarine",
    "Lupo-class_frigate",
    "MEKO#MEKO_200",
    "Maestrale-class_frigate",
    "Makassar-class_landing_platform_dock",
    "Maya-class_destroyer",
    "Minerva-class_corvette",
    "Mistral-class_amphibious_assault_ship",
    "Murasame-class_destroyer_(1994)",
    "Nanuchka-class_corvette",
    "Neustrashimy-class_frigate",
    "Nilgiri-class_frigate_(2019)",
    "Nimitz-class_aircraft_carrier",
    "Niterói-class_frigate",
    "Ohio-class_submarine",
    "Oliver_Hazard_Perry-class_frigate",
    "Oscar-class_submarine",
    "Oyashio-class_submarine",
    "Pohang-class_corvette",
    "Queen_Elizabeth-class_aircraft_carrier",
    "Ropucha-class_landing_ship",
    "Rotterdam-class_landing_platform_dock",
    "Rubis-class_submarine",
    "Sa'ar_5-class_corvette",
    "Sachsen-class_frigate",
    "San_Antonio-class_amphibious_transport_dock",
    "San_Giorgio-class_landing_ship_dock",
    "Scorpène-class_submarine",
    "Seawolf-class_submarine",
    "Sejong_the_Great-class_destroyer",
    "Shivalik-class_frigate",
    "Sierra-class_submarine",
    "Sigma-class_corvette",
    "Skjold-class_corvette",
    "Slava-class_cruiser",
    "Sovremenny-class_destroyer",
    "Spanish_ship_Juan_Carlos_I",
    "Steregushchiy-class_corvette",
    "Suffren-class_submarine",
    "Sōryū-class_submarine",
    "TCG_Anadolu",
    "Taigei-class_submarine",
    "Takanami-class_destroyer",
    "Talwar-class_frigate",
    "Thetis-class_ocean_patrol_vessel",
    "Ticonderoga-class_cruiser",
    "Triomphant-class_submarine",
    "Type_021_frigate",
    "Type_039A_submarine",
    "Type_039_submarine",
    "Type_051B_destroyer",
    "Type_051C_destroyer",
    "Type_052B_destroyer",
    "Type_052C_destroyer",
    "Type_052D_destroyer",
    "Type_052_destroyer",
    "Type_053H3_frigate",
    "Type_054A_frigate",
    "Type_054_frigate",
    "Type_055_destroyer",
    "Type_056_corvette",
    "Type_071_amphibious_transport_dock",
    "Type_075_landing_helicopter_dock",
    "Type_092_submarine",
    "Type_093_submarine",
    "Type_094_submarine",
    "Type_209_submarine",
    "Type_21_frigate",
    "Type_212_submarine",
    "Type_214_submarine",
    "Type_22_frigate",
    "Type_23_frigate",
    "Type_26_frigate",
    "Type_42_destroyer",
    "Type_45_destroyer",
    "Udaloy-class_destroyer",
    "Ula-class_submarine",
    "Ulsan-class_frigate",
    "Upholder-class_submarine",
    "Valour-class_frigate",
    "Vanguard-class_submarine",
    "Vasco_da_Gama-class_frigate",
    "Vikrant-class_aircraft_carrier",
    "Virginia-class_submarine",
    "Visakhapatnam-class_destroyer",
    "Visby-class_corvette",
    "Walrus-class_submarine",
    "Wasp-class_amphibious_assault_ship",
    "Yasen-class_submarine",
    "Yavuz-class_frigate",
    "Yuting-class_tank_landing_ship",
    "Zumwalt-class_destroyer",
    "Álvaro_de_Bazán-class_frigate",
    "Ōsumi-class_tank_landing_ship",
]

HEADERS = {"User-Agent": "yolo-sea-guide/1.0 (annotation tool)"}
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg"}


def scan_existing():
    """Construit le cache depuis les fichiers déjà présents dans images/."""
    found = {}
    for f in IMG_DIR.iterdir():
        if f.suffix.lower() in EXTS:
            slug = f.stem  # nom sans extension = slug wiki
            found[slug] = f"images/{f.name}"
    return found


def api_url(slug):
    return f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(slug)}"


def fetch_with_retry(url, timeout=12, max_retries=5):
    delay = 2.0
    for attempt in range(max_retries):
        try:
            req = Request(url, headers=HEADERS)
            with urlopen(req, timeout=timeout) as r:
                return r.read()
        except HTTPError as e:
            if e.code == 429:
                wait = min(float(e.headers.get("Retry-After", delay)), 5)
                print(f"\n    429 — attente {wait:.0f}s ({attempt + 1}/{max_retries})", end=" ", flush=True)
                time.sleep(wait)
                delay = min(delay * 2, 5)
            else:
                raise
    raise RuntimeError(f"Echec après {max_retries} tentatives")


def fetch_json(url):
    return json.loads(fetch_with_retry(url, timeout=12).decode())


def ext_from_url(url):
    e = url.split("?")[0].rsplit(".", 1)[-1].lower()
    return e if e in ("jpg", "jpeg", "png", "svg", "webp", "gif") else "jpg"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    IMG_DIR.mkdir(exist_ok=True)

    # 1. Charger le cache existent (JSON)
    cache = {}
    if CACHE_FILE.exists():
        cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))

    # 2. Scanner images/ et injecter ce qui est déjà là (manuel ou ancien script)
    on_disk = scan_existing()
    injected = 0
    for slug, rel in on_disk.items():
        if slug not in cache or cache[slug] is None:
            cache[slug] = rel
            injected += 1
    if injected:
        print(f"[scan] {injected} image(s) détectée(s) sur disque ajoutées au cache.")

    slugs = list(dict.fromkeys(WIKI_SLUGS))
    ok = err = skip = 0

    for i, slug in enumerate(slugs, 1):
        rel = cache.get(slug)
        if rel and (IMG_DIR.parent / rel).exists() and not args.force:
            print(f"[{i:>3}/{len(slugs)}] SKIP  {slug}")
            skip += 1
            continue

        print(f"[{i:>3}/{len(slugs)}] GET   {slug}", end=" … ", flush=True)
        try:
            data = fetch_json(api_url(slug))
            img_url = (data.get("originalimage") or {}).get("source") or (data.get("thumbnail") or {}).get("source")

            if not img_url:
                print("pas d'image")
                cache[slug] = None
                err += 1
            else:
                ext = ext_from_url(img_url)
                fname = f"{slug[:80].replace('/', '_')}.{ext}"
                dest = IMG_DIR / fname
                dest.write_bytes(fetch_with_retry(img_url, timeout=20))
                cache[slug] = f"images/{fname}"
                print(f"OK → {fname}")
                ok += 1

        except Exception as e:
            print(f"ERREUR ({e})")
            cache[slug] = None
            err += 1

        time.sleep(1.2)

        if i % 10 == 0:
            CACHE_FILE.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")

    CACHE_FILE.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")

    # Générer img_cache.js — fonctionne en file:// et http://
    entries = ",\n  ".join(f'"{k}": {json.dumps(v)}' for k, v in cache.items() if v is not None)
    CACHE_JS.write_text(
        f"// Généré automatiquement par download_images.py\nwindow.IMG_CACHE = {{\n  {entries}\n}};\n", encoding="utf-8"
    )

    print(f"\n{'=' * 55}")
    print(f"Terminé : {ok} téléchargées · {skip} déjà présentes · {err} sans image")
    print(f"Cache JS : {CACHE_JS}  ({sum(1 for v in cache.values() if v)} images)")


if __name__ == "__main__":
    main()
