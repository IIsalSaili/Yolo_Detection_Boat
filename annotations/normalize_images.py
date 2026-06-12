"""
Renomme les images téléchargées manuellement au format wiki-slug
pour que download_images.py les détecte et ne les re-télécharge pas.

Usage :
    cd annotations
    python normalize_images.py          # aperçu (dry-run)
    python normalize_images.py --apply  # renommage réel
"""

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

IMG_DIR = Path(__file__).parent / "images"

# Mapping : nom actuel → slug wiki cible (sans extension — on la conserve)
# None = fichier ambigu, on le laisse en place (log only)
RENAMES = {
    # ── Korean ships ─────────────────────────────────────────────────────
    "2007년12월3일-함정기동_(7193822678).jpg":
        "Gwanggaeto_the_Great-class_destroyer.jpg",
    "2009년5월15일_해군_1함대훈련_(7193824738).jpg":
        "Chungmugong_Yi_Sun-sin-class_destroyer.jpg",
    "20130626_대한해협_전승행사_(4)_(9460603621).jpg":
        "Sejong_the_Great-class_destroyer.jpg",          # Korea Strait ceremony → Sejong

    # ── Russian ships ─────────────────────────────────────────────────────
    "AdmiralVinogradov2009.jpg":
        "Udaloy-class_destroyer.jpg",
    "Corvette_Steregushchiy.jpg":
        "Steregushchiy-class_corvette.jpg",
    "Sderzhannyy1980.jpg":
        "Kashin-class_destroyer.jpg",
    "«Александр_Невский»_в_Вилючинске.jpg":
        "Borei-class_submarine.jpg",

    # ── German ships ──────────────────────────────────────────────────────
    "Bm_F216.jpg":
        "Brandenburg-class_frigate.jpg",                 # F216 Schleswig-Holstein = Brandenburg
    "F221_FGS_Hessen_(8640310899).jpg":
        "Sachsen-class_frigate.jpg",                     # F221 Hessen = Sachsen (F124)

    # ── Chinese ships ─────────────────────────────────────────────────────
    "Chinese_Navy_Guided-Missile_Destroyer_Xian_(153)_Departs_Joint_Base_Pearl_Harbor-Hickam_Following_the_Conclusion_of_RIMPAC_2016_160804-N-IU636-106.jpg":
        "Type_052B_destroyer.jpg",                       # DDG-153 Xian = Type 052B
    "Guangzhou_168_(cropped).jpg":
        "Type_052C_destroyer.jpg",                       # DDG-168 Guangzhou = Type 052C
    "PLANS_Shenyang_(DDG-115)_20150825_2.jpg":
        "Type_052C_destroyer.jpg",                       # DDG-115 Shenyang = 052C (doublon → voir logique ci-dessous)
    "Song-class_Submarine_5.jpg":
        "Type_039_submarine.jpg",

    # ── British ships ─────────────────────────────────────────────────────
    "HMS_Ambush_long.jpg":
        "Astute-class_submarine.jpg",
    "HMS_Birmingham_D86.jpg":
        "Type_42_destroyer.jpg",
    "HMS_Daring-1.jpg":
        "Type_45_destroyer.jpg",

    # ── French ships ──────────────────────────────────────────────────────
    "French_destroyer_Forbin_(D620)_underway_in_the_Arabian_Sea_on_31_May_2009_(090531-N-9988F-406).jpg":
        "Horizon-class_frigate.jpg",
    "Sous-marin-Flore.jpg":
        "Daphné-class_submarine.jpg",

    # ── Norwegian ships ───────────────────────────────────────────────────
    "HNoMS_Otto_Sverdrup.jpg":
        "Fridtjof_Nansen-class_frigate.jpg",

    # ── Indian ships ──────────────────────────────────────────────────────
    "INS_Kalvari_(S21)_at_sea.jpg":
        "Scorpène-class_submarine.jpg",

    # ── Italian ships ─────────────────────────────────────────────────────
    "ITS_Giuseppe_Garibaldi_(C_551).jpg":
        "Italian_aircraft_carrier_Giuseppe_Garibaldi.jpg",

    # ── Japanese ships ────────────────────────────────────────────────────
    "DDG-178_04l.jpg":
        "Atago-class_destroyer.jpg",                     # DDG-178 Ashigara = Atago
    "DDG_Maya.jpg":
        "Maya-class_destroyer.jpg",
    "JDS_Hatakaze.jpg":
        "Hatakaze-class_destroyer.jpg",
    "JDS_Makinami_(DD_112).jpg":
        "Takanami-class_destroyer.jpg",                  # DD-112 Makinami = Takanami
    "JS_Myōkō(DDG-175).jpg":
        "Kongō-class_destroyer.jpg",                     # DDG-175 Myōkō = Kongō
    "JS_Umigiri（DD-158）.jpg":
        "Asagiri-class_destroyer.jpg",                   # DD-158 Umigiri = Asagiri
    "Japanese_Submarine_Oyashio_SS590.jpg":
        "Oyashio-class_submarine.jpg",

    # ── Korean ships ─────────────────────────────────────────────────────
    "ROKS_Yi_SunShin_DDH-975.jpg":
        "Chungmugong_Yi_Sun-sin-class_destroyer.jpg",

    # ── Malaysian ships ───────────────────────────────────────────────────
    "KD_Lekiu(FFG30)_and_KD_Jebat(FFG29)_with_USS_George_Washington_(CVN_73).jpeg":
        "Lekiu-class_frigate.jpeg",

    # ── Italian ships ─────────────────────────────────────────────────────
    "Maestrale-Class_MAESTRALE_(F_570).jpg":
        "Maestrale-class_frigate.jpg",

    # ── Misc / long names ─────────────────────────────────────────────────
    "Durand_de_la_Penne_D560.jpg":
        "Durand_de_la_Penne-class_destroyer.jpg",
    "Submarine_Delta_IV_class.jpg":
        "Delta-class_submarine.jpg",
    "Swedish_attack_submarine_HMS_Gotland.jpg":
        "Gotland-class_submarine.jpg",
    "USS_Connecticut_SSN_22.jpg":
        "Seawolf-class_submarine.jpg",
    "USS_Scott_DDG-995.jpg":
        "Kidd-class_destroyer.jpg",                      # DDG-995 Scott = Kidd class
    "USS_Wasp_(LHD-1)_Osprey_2.jpg":
        "Wasp-class_amphibious_assault_ship.jpg",
    "US_Navy_040730-N-1234E-002_PCU_Virginia_(SSN_774)_returns_to_the_General_Dynamics_Electric_Boat_shipyard.jpg":
        "Virginia-class_submarine.jpg",
    "US_Navy_080923-N-2183K-024_The_guided-missile_cruiser_USS_Cape_St._George_(CG_71)_steams_in_the_Indian_Ocean_supporting_maritime_security_operations_in_the_U.S._5th_Flee.jpg":
        None,                                            # Ticonderoga déjà présent — doublon, supprimé
    "US_Navy_100326-N-4774B-247_The_Peru_navy_frigate_BAP_Carvajal_(FM_51)_is_participating_in_Southern_Seas_2010,_a_U.S._Southern_Command-directed_operation_that_provides_U.jpg":
        "Lupo-class_frigate.jpg",
    "U.S._Navy_guided-missile_frigate_FFG(X)_artist_rendering,_30_April_2020_(200430-N-NO101-150).jpg":
        "Constellation-class_frigate.jpg",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Effectuer les renommages (sans ce flag : dry-run)")
    args = parser.parse_args()

    dry = not args.apply
    if dry:
        print("DRY-RUN — aucun fichier modifié. Relancer avec --apply pour appliquer.\n")

    renamed = skipped = deleted = errors = 0

    for src_name, dst_name in RENAMES.items():
        src = IMG_DIR / src_name

        if not src.exists():
            print(f"  ABSENT   {src_name}")
            errors += 1
            continue

        if dst_name is None:
            # Doublon inutile → supprimer
            print(f"  SUPPRIME {src_name}")
            if not dry:
                src.unlink()
            deleted += 1
            continue

        dst = IMG_DIR / dst_name

        if dst.exists():
            # La cible existe déjà → garder la plus grande (meilleure qualité) et supprimer l'autre
            if src.stat().st_size >= dst.stat().st_size:
                print(f"  REMPLACE {dst_name}  ← {src_name}  (source plus grosse)")
                if not dry:
                    dst.unlink()
                    src.rename(dst)
                renamed += 1
            else:
                print(f"  DOUBLON  {src_name}  (cible déjà présente et plus grosse → source supprimée)")
                if not dry:
                    src.unlink()
                skipped += 1
        else:
            print(f"  RENAME   {src_name}\n           → {dst_name}")
            if not dry:
                src.rename(dst)
            renamed += 1

    print(f"\n{'='*55}")
    print(f"{'[DRY-RUN] ' if dry else ''}Renommés : {renamed}  Doublons ignorés : {skipped}  Supprimés : {deleted}  Absents : {errors}")
    if dry:
        print("Relancer avec --apply pour appliquer.")


if __name__ == "__main__":
    main()
