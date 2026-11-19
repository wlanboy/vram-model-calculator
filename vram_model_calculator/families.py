"""Generates the model family navigation in index.html from the pages in
modell-familien/. The link text is the page's <meta name="nav-label">, or
its <title> up to the first " – "."""
import html
import os
import re

FAMILY_DIR = "modell-familien"
INDEX_FILE = "index.html"

NAV_RE = re.compile(r"(<!-- BEGIN FAMILIES[^>]*-->\n)(.*?)(\s*<!-- END FAMILIES -->)", re.DOTALL)
LABEL_RE = re.compile(r'<meta\s+name="nav-label"\s+content="([^"]*)"', re.IGNORECASE)
TITLE_RE = re.compile(r"<title>(.*?)</title>", re.IGNORECASE | re.DOTALL)


def family_label(page_html, fallback):
    m = LABEL_RE.search(page_html)
    if m:
        return html.unescape(m.group(1)).strip()
    m = TITLE_RE.search(page_html)
    if m:
        return html.unescape(m.group(1)).split(" – ")[0].strip()
    return fallback


def collect_families(root="."):
    """[(label, relative href)] sorted by label."""
    family_dir = os.path.join(root, FAMILY_DIR)
    families = []
    for name in os.listdir(family_dir):
        if not name.endswith(".html"):
            continue
        with open(os.path.join(family_dir, name), encoding="utf-8") as f:
            families.append((family_label(f.read(), name[:-5]), f"{FAMILY_DIR}/{name}"))
    return sorted(families, key=lambda f: f[0].lower())


def render_nav(families):
    return "\n".join(
        f'    <a href="{html.escape(href)}" class="family-link">{html.escape(label)}</a>'
        for label, href in families
    )


def generate(root=".", check=False):
    """Rewrites the block between the FAMILIES markers in index.html.
    Returns True if the file changed (or would change, with check=True)."""
    index_path = os.path.join(root, INDEX_FILE)
    with open(index_path, encoding="utf-8") as f:
        old = f.read()
    if not NAV_RE.search(old):
        raise ValueError(f"Marker <!-- BEGIN FAMILIES --> / <!-- END FAMILIES --> fehlen in {index_path}")
    nav = render_nav(collect_families(root))
    new = NAV_RE.sub(lambda m: m.group(1) + nav + m.group(3), old, count=1)
    if new == old:
        return False
    if not check:
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(new)
    return True


def add_arguments(parser):
    parser.add_argument("--root", default=".", help="Repo-Verzeichnis mit index.html (Default: %(default)s)")
    parser.add_argument("--check", action="store_true", help="Nichts schreiben, Exit-Code 1 bei veralteter Navigation")


def run(args):
    changed = generate(args.root, check=args.check)
    if args.check:
        print("❌ Navigation veraltet" if changed else "✅ Navigation aktuell")
        return 1 if changed else 0
    print("✅ Navigation aktualisiert" if changed else "✅ Navigation war schon aktuell")
    return 0
