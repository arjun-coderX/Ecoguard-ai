"""Inline engine.js + dashboard_data.json into a single HTML file.

    python scripts/build_dashboard.py

Writes outputs/vanagni_dashboard.html (open it directly in a browser,
no server needed) and outputs/vanagni_fragment.html (body-only, for
embedding in a host page).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    tpl = (ROOT / "dashboard" / "template.html").read_text(encoding="utf-8")
    engine = (ROOT / "dashboard" / "engine.js").read_text(encoding="utf-8")
    data = (ROOT / "outputs" / "dashboard_data.json").read_text(encoding="utf-8")
    data = data.replace("</", "<\\/")          # never close the script tag early
    frag = tpl.replace("/*__ENGINE__*/", engine).replace("__DATA__", data)
    (ROOT / "outputs" / "vanagni_fragment.html").write_text(frag, encoding="utf-8")
    page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            '</head>\n<body>\n' + frag + '\n</body>\n</html>\n')
    out = ROOT / "outputs" / "vanagni_dashboard.html"
    out.write_text(page, encoding="utf-8")
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "model.html").write_text(page, encoding="utf-8")   # GitHub Pages: full model dashboard (index.html is the EcoGuard page)
    print(f"wrote {out} and docs/model.html ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
