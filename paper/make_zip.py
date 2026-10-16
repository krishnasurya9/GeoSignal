"""
paper/make_zip.py
Creates paper/geosignal_submission.zip for drag-and-drop onto Overleaf.

Contents:
  draft_manuscript.tex
  figures/cs01_propagation_map.png
  ...
  figures/cs06_propagation_map.png
"""
import zipfile
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[1]
TEX_SRC = ROOT / "paper" / "draft_manuscript.tex"
FIG_DIR = ROOT / "paper" / "figures"
OUT_ZIP = ROOT / "paper" / "geosignal_submission.zip"

tex = TEX_SRC.read_text(encoding="utf-8")

for i in range(1, 7):
    for old in (
        f"../results/figures/cs0{i}/propagation_map_sm.png",
        f"../results/figures/cs0{i}/propagation_map.png",
        f"figures/cs0{i}_propagation_map_sm.png",
    ):
        tex = tex.replace(old, f"figures/cs0{i}_propagation_map.png")

with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("draft_manuscript.tex", tex.encode("utf-8"))
    print(f"\n  + draft_manuscript.tex  ({len(tex)//1024} KB)")
    for i in range(1, 7):
        name = f"cs0{i}_propagation_map.png"
        png = FIG_DIR / name
        if not png.exists():
            png = ROOT / "results" / "figures" / f"cs0{i}" / "propagation_map.png"
        if png.exists():
            zf.write(str(png), f"figures/{name}")
            print(f"  + figures/{name}  ({png.stat().st_size // 1024} KB)")
        else:
            print(f"  MISSING: {name}")

print(f"\nZIP created: {OUT_ZIP}")
print(f"ZIP size:    {OUT_ZIP.stat().st_size // 1024} KB\n")

print("Contents:")
with zipfile.ZipFile(OUT_ZIP) as zf:
    for info in zf.infolist():
        print(f"  {info.filename:<55}  {info.file_size // 1024:>5} KB")
