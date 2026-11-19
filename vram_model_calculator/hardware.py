"""Generates the MODELS tables of hardware/<device>/index.html from
hardware/<device>/device.json and models_cache.json.

device.json holds the device profile and the hand-measured values (speed,
rating). For every model that can be matched to a cache entry (explicit
cache_key, or same normalized name and quant), VRAM and context come from
the cache via vram_calculator.estimate, so the pages follow the same formula
as the calculator. Unmatched models keep the VRAM value stored in device.json.
"""
import ast
import json
import os
import re
import statistics

from .config import resolve_cache_file
from .vram_calculator import (
    DEFAULT_OVERHEAD_GB,
    KV_CACHE_TYPE_DEFAULT,
    estimate,
    load_models,
)

DEVICE_FILE = "device.json"
PAGE_FILE = "index.html"
OVERVIEW_FILE = "index.html"
DEFAULT_DEVICE_CTX = 8192

MODELS_RE = re.compile(r"const MODELS = \[.*?\n\];", re.DOTALL)
H1_RE = re.compile(r"<h1>(.*?)</h1>", re.DOTALL)

# Statuses counted on the overview cards.
STAT_GREAT = "great"
STAT_WELL = "well"


def normalize(text):
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def match_cache_entry(model, cache):
    """Cache entry for a device.json model: its cache_key, or the single
    entry with the same normalized name and quant. None if not unique."""
    key = model.get("cache_key")
    if key:
        return cache.get(key)
    name, quant = normalize(model.get("name")), normalize(model.get("quant"))
    hits = [e for e in cache.values() if normalize(e.get("name")) == name and normalize(e.get("quant")) == quant]
    return hits[0] if len(hits) == 1 else None


def _ctx_columns(n_ctx):
    k = round(n_ctx / 1000)
    return f"{k}K", k


def build_rows(device, cache):
    """MODELS rows in the schema hardware.js expects. Returns (rows, matched)."""
    vram_gb = device["vram_gb"]
    ctx = device.get("ctx", DEFAULT_DEVICE_CTX)
    est_kwargs = {
        "cache_type": device.get("cache_type", KV_CACHE_TYPE_DEFAULT),
        "overhead_gb": device.get("overhead_gb", DEFAULT_OVERHEAD_GB),
    }
    rows = []
    matched = 0
    for m in device["models"]:
        vram, ctx_label, ctx_num = m["vram_gb"], m["ctx_label"], m["ctx_num"]
        entry = match_cache_entry(m, cache)
        if entry:
            matched += 1
            vram = round(estimate(entry, ctx, **est_kwargs).total_gb, 1)
            if entry.get("n_ctx_orig"):
                ctx_label, ctx_num = _ctx_columns(entry["n_ctx_orig"])
        rows.append([
            m["name"], m["params_b"], m["type"], m["quant"], vram, round(vram / vram_gb * 100),
            m["speed_ts"], ctx_label, ctx_num, m["status"], m["status_text"], m["grade"], m["score"],
        ])
    return rows, matched


def render_models_block(rows):
    lines = ",\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    return "const MODELS = [\n" + lines + ("," if rows else "") + "\n];"


def device_from_page(html):
    """Builds a device.json skeleton from an existing hand-written page
    (one-time migration, `hardware --init`)."""
    block = MODELS_RE.search(html)
    if not block:
        raise ValueError("kein MODELS-Array gefunden")
    keys = ["name", "params_b", "type", "quant", "vram_gb", "vram_pct", "speed_ts", "ctx_label", "ctx_num",
            "status", "status_text", "grade", "score"]
    # One row per line; a // comment line names the source of the rows after it.
    models = []
    source = None
    for line in block.group(0).splitlines()[1:-1]:
        line = line.strip()
        if line.startswith("//"):
            source = line[2:].strip()
        elif line:
            model = dict(zip(keys, ast.literal_eval(line.rstrip(","))))
            if source:
                model["source"] = source
            models.append(model)
    capacities = [m["vram_gb"] / m["vram_pct"] * 100 for m in models if m["vram_pct"]]
    for m in models:
        del m["vram_pct"]
    title = H1_RE.search(html)
    return {
        "title": title.group(1).strip() if title else "",
        "vram_gb": round(statistics.median(capacities)) if capacities else 0,
        "ctx": DEFAULT_DEVICE_CTX,
        "models": models,
    }


def _card_re(device_dir):
    return re.compile(
        r'(<a class="hardware-card" href="' + re.escape(device_dir) + r'/index.html">.*?'
        r'<div class="hardware-card-stats">)(.*?)(</div>)',
        re.DOTALL,
    )


def render_card_stats(rows):
    great = sum(1 for r in rows if r[9] == STAT_GREAT)
    well = sum(1 for r in rows if r[9] == STAT_WELL)
    return (
        f'\n      <span class="hardware-card-stat">{len(rows)} Modelle</span>'
        f'\n      <span class="hardware-card-stat stat-great">{great} exzellent</span>'
        f'\n      <span class="hardware-card-stat stat-well">{well} gut</span>\n    '
    )


def _replace_models(page, rows):
    block = render_models_block(rows)
    return MODELS_RE.sub(lambda _: block, page, count=1)


def _replace_card_stats(overview, device_dir, rows):
    stats = render_card_stats(rows)
    return _card_re(device_dir).sub(lambda m: m.group(1) + stats + m.group(3), overview)


def _write_if_changed(path, new, check):
    with open(path, encoding="utf-8") as f:
        old = f.read()
    if old == new:
        return False
    if not check:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new)
    return True


def generate(hardware_dir="hardware", cache_file=None, init=False, check=False):
    """Regenerates all device pages and the overview cards. With check=True
    nothing is written; returns the list of files that would change."""
    cache = load_models(resolve_cache_file(cache_file)) or {}
    changed = []
    overview_path = os.path.join(hardware_dir, OVERVIEW_FILE)
    with open(overview_path, encoding="utf-8") as f:
        overview = f.read()

    for device_dir in sorted(os.listdir(hardware_dir)):
        page_path = os.path.join(hardware_dir, device_dir, PAGE_FILE)
        device_path = os.path.join(hardware_dir, device_dir, DEVICE_FILE)
        if not os.path.isfile(page_path):
            continue
        with open(page_path, encoding="utf-8") as f:
            page = f.read()

        if not os.path.isfile(device_path):
            if not init:
                print(f"⏭️ {device_dir}: keine {DEVICE_FILE} (anlegen mit --init)")
                continue
            device = device_from_page(page)
            if not check:
                with open(device_path, "w", encoding="utf-8") as f:
                    json.dump(device, f, indent=2, ensure_ascii=False)
                    f.write("\n")
            changed.append(device_path)
            print(f"🆕 {device_path} aus der bestehenden Seite angelegt ({len(device['models'])} Modelle)")
        else:
            with open(device_path, encoding="utf-8") as f:
                device = json.load(f)

        rows, matched = build_rows(device, cache)
        new_page = _replace_models(page, rows)
        if _write_if_changed(page_path, new_page, check):
            changed.append(page_path)
        print(f"🖥️ {device_dir}: {len(rows)} Modelle, {matched} aus dem Cache berechnet")

        overview = _replace_card_stats(overview, device_dir, rows)

    if _write_if_changed(overview_path, overview, check):
        changed.append(overview_path)
    return changed


def add_arguments(parser):
    parser.add_argument("--dir", default="hardware", help="Verzeichnis mit den Geräte-Seiten (Default: %(default)s)")
    parser.add_argument("--init", action="store_true", help="Fehlende device.json aus der bestehenden Seite anlegen")
    parser.add_argument("--check", action="store_true", help="Nichts schreiben, Exit-Code 1 bei veralteten Seiten")
    parser.add_argument("--cache", metavar="PATH", default=None, help="Pfad zu models_cache.json")


def run(args):
    changed = generate(args.dir, args.cache, init=args.init, check=args.check)
    verb = "veraltet" if args.check else "aktualisiert"
    for path in changed:
        print(f"  {verb}: {path}")
    return 1 if args.check and changed else 0
