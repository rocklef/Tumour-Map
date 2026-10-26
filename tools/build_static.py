"""Build the read-only online demo (for Vercel) from cases already processed by the local server.

Usage (from tumour-map/): python tools/build_static.py
Takes the demo patients named in backend/seed.json, with their finished studies, and writes site/.
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
db = json.loads((ROOT / "data" / "db.json").read_text())
seed = json.loads((ROOT / "backend" / "seed.json").read_text())
demo_mrns = {p["mrn"] for p in seed["demo_patients"]}
demo_labels = {p["study_label"] for p in seed["demo_patients"]}

if SITE.exists():
    shutil.rmtree(SITE)
(SITE / "demo" / "studies").mkdir(parents=True)

html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
html = html.replace('<meta charset="utf-8">', '<meta charset="utf-8">\n<meta name="tm-static" content="1">', 1)
(SITE / "index.html").write_text(html, encoding="utf-8")

patients = []
for p in sorted(db["patients"], key=lambda p: -p["created"]):
    if p["mrn"] not in demo_mrns:
        continue
    studies = [s for s in db["studies"] if s["patient"] == p["id"] and s["status"] == "done" and s["label"] in demo_labels]
    if not studies:
        raise SystemExit(f"{p['name']} has no finished study yet. Let the local server finish processing first.")
    patients.append({**p, "studies": sorted(studies, key=lambda s: -s["created"])})
    for s in studies:
        src = ROOT / "data" / "studies" / s["id"]
        out = SITE / "demo" / "studies" / s["id"]
        out.mkdir()
        report = json.loads((src / "report.json").read_text())
        (SITE / "demo" / "studies" / f"{s['id']}.json").write_text(json.dumps({**s, "report": report, "patient": p}))
        for f in src.glob("*.bin"):
            name = f.name.replace("vol_", "")
            shutil.copy(f, out / name)
(SITE / "demo" / "patients.json").write_text(json.dumps(patients))
(SITE / "vercel.json").write_text(json.dumps({"headers": [{"source": "/demo/(.*)", "headers": [{"key": "Cache-Control", "value": "public, max-age=86400"}]}]}))

total = sum(f.stat().st_size for f in SITE.rglob("*") if f.is_file())
print(f"site/ built: {len(patients)} patients, {total / 1e6:.1f} MB")
