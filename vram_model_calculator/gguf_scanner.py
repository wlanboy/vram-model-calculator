import json
import os
import re
import sys

from tqdm import tqdm

from ._model import METADATA_DUMP_FILE, NotAnLLMError, clean_name, get_model_params

CACHE_FILE = "models_cache.json"
SHARD_RE = re.compile(r'-(\d{5})-of-(\d{5})\.gguf$', re.IGNORECASE)

# Well-known model locations, scanned automatically when no path is given
# on the command line. Missing entries (e.g. an unmounted /wdblack) are
# skipped by update_cache() without error.
DEFAULT_DIRS = [
    os.path.expanduser("~/LMStudio/models/"),
    os.path.expanduser("~/.lmstudio/models/"),
    os.path.expanduser("~/.cache/huggingface/hub/"),
    "/models",
    "/wdblack/models",
]

# Path substrings (case-insensitive) that mark a model as excluded from
# scanning entirely, e.g. "heretic"-style abliterated/uncensored finetunes.
SKIP_PATH_SUBSTRINGS = ["heretic"]


def get_shard_info(path):
    """Returns (shard_index, total_shards) or None if not a shard file."""
    m = SHARD_RE.search(os.path.basename(path))
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _migrate_key(key):
    """Strip org/user prefix from cache key if not already starting with a GGUF model folder."""
    parts = key.split('/')
    if len(parts) >= 2 and not parts[0].upper().endswith('-GGUF'):
        return '/'.join(parts[1:])
    return key


def _migrate_cache(cache):
    """Normalize old-format keys with org prefix to model-relative keys, removing duplicates."""
    migrated = {}
    changed = 0
    for key, entry in cache.items():
        new_key = _migrate_key(key)
        if new_key != key:
            changed += 1
            if isinstance(entry, dict):
                entry = dict(entry)
                entry['rel_path'] = new_key
        if new_key not in migrated:
            migrated[new_key] = entry
    removed = len(cache) - len(migrated)
    if changed:
        print(f"♻️ {changed} Einträge migriert, {removed} Duplikate entfernt.")
    return migrated


def _refresh_names(cache):
    """Re-apply clean_name to all cached entries so improved rules take effect without a rescan."""
    refreshed = 0
    for key, entry in cache.items():
        if not isinstance(entry, dict) or 'name' not in entry:
            continue
        old = entry['name']
        new = clean_name(old)
        if new and new != old:
            cache[key] = {**entry, 'name': new}
            refreshed += 1
    if refreshed:
        print(f"✨ {refreshed} Modellnamen aktualisiert.")
    return cache


def needs_scan(rel_key, abs_path, cache):
    if rel_key not in cache:
        return True
    entry = cache[rel_key]
    if entry.get("file_size_bytes") != os.path.getsize(abs_path):
        return True
    return entry.get("has_missing_fields", False)


def update_cache(base_dirs):
    if isinstance(base_dirs, str):
        base_dirs = [base_dirs]

    base_dirs = [os.path.abspath(d) for d in base_dirs]

    cache = {}
    file_version = 0

    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                loaded = json.load(f)
            file_version = loaded.get("_version", 0)
            raw = {k: v for k, v in loaded.items() if k != "_version"}
            cache = _refresh_names(_migrate_cache(raw))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError) as e:
            print(f"⚠️ Cache-Datei korrupt, erstelle neu. ({e})")

    all_pairs = []
    skipped_by_filter = 0
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            print(f"❌ Pfad nicht gefunden: {base_dir}")
            continue
        print(f"📂 Scanne: {base_dir}")
        for root, _, files in os.walk(base_dir):
            for f in files:
                if not f.endswith(".gguf"):
                    continue
                abs_path = os.path.join(root, f)
                if any(s in abs_path.lower() for s in SKIP_PATH_SUBSTRINGS):
                    skipped_by_filter += 1
                    continue
                all_pairs.append((abs_path, base_dir))

    if skipped_by_filter:
        print(f"⏭️ {skipped_by_filter} Datei(en) durch Filter übersprungen ({', '.join(SKIP_PATH_SUBSTRINGS)}).")

    if not all_pairs:
        print("❌ Keine GGUF-Dateien in den angegebenen Verzeichnissen gefunden.")
        return cache

    # Group shard files: only process first shard, aggregate total size
    shard_groups = {}
    non_shard_pairs = []
    for abs_path, base_dir in all_pairs:
        info = get_shard_info(abs_path)
        if info:
            base = SHARD_RE.sub('.gguf', abs_path)
            shard_groups.setdefault(base, []).append((info[0], abs_path, base_dir))
        else:
            non_shard_pairs.append((abs_path, base_dir))

    shard_meta = {}
    representative_pairs = list(non_shard_pairs)
    for base, shards in shard_groups.items():
        shards.sort(key=lambda x: x[0])
        first_path, first_base = shards[0][1], shards[0][2]
        total_size = sum(os.path.getsize(p) for _, p, _ in shards)
        shard_meta[first_path] = (total_size, len(shards))
        representative_pairs.append((first_path, first_base))

    new_pairs = [
        (p, bd) for p, bd in representative_pairs
        if needs_scan(_migrate_key(os.path.relpath(p, bd)), p, cache)
    ]

    if not new_pairs:
        print("✅ Alles aktuell. Keine neuen oder geänderten GGUF-Dateien gefunden.")
        return cache

    print(f"🔍 {len(new_pairs)} Modelle werden analysiert...")

    errors = []
    skipped = []
    if os.path.exists(METADATA_DUMP_FILE):
        open(METADATA_DUMP_FILE, 'w').close()

    for abs_path, base_dir in tqdm(new_pairs, desc="GGUF Scan", unit="file", colour="green"):
        rel = _migrate_key(os.path.relpath(abs_path, base_dir))
        try:
            if abs_path in shard_meta:
                total_size, num_shards = shard_meta[abs_path]
                params = get_model_params(abs_path, file_size_bytes=total_size)
                params["num_shards"] = num_shards
            else:
                params = get_model_params(abs_path)
            params["rel_path"] = rel
            cache[rel] = params
        except NotAnLLMError as e:
            skipped.append(f"Datei: {rel} | Grund: {e}")
        except Exception as e:  # noqa: BLE001 -- one bad/corrupt GGUF file must not abort the whole batch scan
            errors.append(f"Datei: {rel} | Grund: {e}")

    new_version = file_version + 1
    with open(CACHE_FILE, 'w') as f:
        json.dump({"_version": new_version, **dict(sorted(cache.items()))}, f, indent=4)

    if skipped:
        print(f"\n⏭️ {len(skipped)} Datei(en) übersprungen (kein LLM):")
        for s in skipped:
            print(f"  - {s}")

    if errors:
        print("\n⚠️ SCAN-FEHLER:")
        for err in errors:
            print(f"  - {err}")

    dump_count = sum(1 for v in cache.values() if isinstance(v, dict) and v.get("has_missing_fields"))
    if dump_count:
        print(f"\n📄 {dump_count} Modell(e) mit fehlenden Feldern → Details in '{METADATA_DUMP_FILE}'")

    print(f"\n💾 Cache v{new_version} gespeichert unter '{CACHE_FILE}' ({len(cache)} Einträge).")
    return cache


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_DIRS
    update_cache(targets)


if __name__ == "__main__":
    main()
