import argparse
import json
import os
import re
from collections import namedtuple

from tqdm import tqdm

from ._model import (
    METADATA_DUMP_FILE,
    MODEL_TYPE_LLM,
    ModelShape,
    NotAnLLMError,
    clean_name,
    get_model_params,
)

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
SKIP_PATH_SUBSTRINGS = ["heretic", "heresy", "uncensored", "abliterated", "huihui", "dolphin"]

# One model to scan: a single file, or the first shard of a complete shard
# set (size is then the sum of all shards).
Candidate = namedtuple("Candidate", "path base_dir size num_shards")


def get_shard_info(path):
    """Returns (shard_index, total_shards) or None if not a shard file."""
    m = SHARD_RE.search(os.path.basename(path))
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _short_key(rel_path):
    """Cache key for a path relative to its base dir: the org/user folder is
    stripped unless the path already starts with a GGUF model folder or an
    HF-cache repo folder (models--org--repo, which carries the model name).
    Must only be applied once, to the full relative path."""
    parts = rel_path.split('/')
    if len(parts) >= 2 and not parts[0].upper().endswith('-GGUF') and not parts[0].startswith('models--'):
        return '/'.join(parts[1:])
    return rel_path


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


def _backfill_kv_bytes(cache):
    """Adds kv_bytes_per_ctx_token to entries scanned before that field
    existed, computed from the dimensions already stored in the cache (no
    GGUF file access needed, so no rescan is required)."""
    backfilled = 0
    for key, entry in cache.items():
        if not isinstance(entry, dict) or entry.get("type") != MODEL_TYPE_LLM:
            continue
        if "kv_bytes_per_ctx_token" in entry or not entry.get("n_layers") or not entry.get("n_embd"):
            continue
        shape = ModelShape(
            n_layers=entry["n_layers"],
            n_embd=entry["n_embd"],
            n_heads=entry.get("n_heads") or 0,
            n_kv_heads=entry.get("n_kv_heads"),
            mla_kv_dim=entry.get("mla_kv_dim"),
        )
        cache[key] = {**entry, "kv_bytes_per_ctx_token": shape.kv_bytes_per_ctx_token()}
        backfilled += 1
    if backfilled:
        print(f"🧮 {backfilled} Cache-Einträge um kv_bytes_per_ctx_token ergänzt.")
    return cache


def needs_scan(rel_key, file_size_bytes, cache):
    """file_size_bytes is the total size of the model (sum of all shards)."""
    entry = cache.get(rel_key)
    if not isinstance(entry, dict):
        return True
    if entry.get("file_size_bytes") != file_size_bytes:
        return True
    if entry.get("has_missing_fields", False):
        return True
    # Entries scanned before per-layer KV support (key/value_length head
    # dims, per-layer KV heads, hybrid attention layers) lack this key;
    # rescanning (once) picks it up without needing a cache wipe.
    return entry.get("type") == MODEL_TYPE_LLM and "head_dim_k" not in entry


def _load_cache():
    """Returns (raw_entries, version); empty on missing or corrupt file."""
    if not os.path.exists(CACHE_FILE):
        return {}, 0
    try:
        with open(CACHE_FILE, 'r') as f:
            loaded = json.load(f)
        version = loaded.get("_version", 0)
        return {k: v for k, v in loaded.items() if k != "_version"}, version
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError) as e:
        print(f"⚠️ Cache-Datei korrupt, erstelle neu. ({e})")
        return {}, 0


def _save_cache(cache, version):
    """Writes atomically: an interrupted write must not destroy the existing cache."""
    tmp_file = CACHE_FILE + ".tmp"
    with open(tmp_file, 'w') as f:
        json.dump({"_version": version, **dict(sorted(cache.items()))}, f, indent=4)
    os.replace(tmp_file, CACHE_FILE)


def _collect_gguf_files(base_dirs):
    """Returns ([(abs_path, base_dir)], skipped_by_filter_count)."""
    pairs = []
    skipped_by_filter = 0
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            print(f"❌ Pfad nicht gefunden: {base_dir}")
            continue
        print(f"📂 Scanne: {base_dir}")
        for root, _, files in os.walk(base_dir):
            for f in files:
                if not f.lower().endswith(".gguf"):
                    continue
                abs_path = os.path.join(root, f)
                # Only the path below base_dir counts, so a base dir like
                # /data/dolphin/ doesn't exclude everything inside it.
                rel_path = os.path.relpath(abs_path, base_dir).lower()
                if any(s in rel_path for s in SKIP_PATH_SUBSTRINGS):
                    skipped_by_filter += 1
                    continue
                pairs.append((abs_path, base_dir))
    return pairs, skipped_by_filter


def _build_candidates(pairs, errors):
    """Groups shard files into one candidate per complete shard set. Incomplete
    sets and unreadable files (e.g. broken symlinks) are reported in errors."""
    candidates = []
    shard_groups = {}
    for abs_path, base_dir in pairs:
        info = get_shard_info(abs_path)
        if info:
            group = SHARD_RE.sub('.gguf', abs_path)
            shard_groups.setdefault(group, []).append((info, abs_path, base_dir))
            continue
        try:
            candidates.append(Candidate(abs_path, base_dir, os.path.getsize(abs_path), None))
        except OSError as e:
            errors.append(f"Datei: {abs_path} | Grund: {e}")

    for group, shards in shard_groups.items():
        shards.sort(key=lambda x: x[0][0])
        indices = [info[0] for info, _, _ in shards]
        totals = {info[1] for info, _, _ in shards}
        expected = max(totals)
        if len(totals) != 1 or indices != list(range(1, expected + 1)):
            errors.append(
                f"Datei: {group} | Grund: unvollständiger Shard-Satz "
                f"(gefunden {indices}, erwartet 1-{expected})"
            )
            continue
        try:
            total_size = sum(os.path.getsize(p) for _, p, _ in shards)
        except OSError as e:
            errors.append(f"Datei: {group} | Grund: {e}")
            continue
        _, first_path, first_base = shards[0]
        candidates.append(Candidate(first_path, first_base, total_size, expected))
    return candidates


def _assign_keys(candidates):
    """Maps cache key -> candidate. Shortened keys (see _short_key) can
    collide; identical copies (same path below different base dirs) are
    dropped, distinct files fall back to their full relative path.
    Returns (keyed, duplicate_paths)."""
    by_key = {}
    for c in candidates:
        by_key.setdefault(_short_key(os.path.relpath(c.path, c.base_dir)), []).append(c)

    keyed = {}
    duplicates = []
    for key, group in by_key.items():
        unique = {}
        for c in group:
            full = os.path.relpath(c.path, c.base_dir)
            if full in unique:
                duplicates.append(c.path)
            else:
                unique[full] = c
        if len(unique) == 1:
            keyed[key] = next(iter(unique.values()))
            continue
        for full, c in unique.items():
            if full in keyed:
                duplicates.append(c.path)
            else:
                keyed[full] = c
    return keyed, duplicates


def update_cache(base_dirs, prune=False):
    """Scans base_dirs and updates the cache. With prune=True, cache entries
    whose file wasn't found in this run are removed. Only use it when all
    model directories are reachable, an unmounted disk looks like deleted models."""
    if isinstance(base_dirs, str):
        base_dirs = [base_dirs]

    base_dirs = [os.path.abspath(d) for d in base_dirs]

    raw, file_version = _load_cache()
    cache = _backfill_kv_bytes(_refresh_names(dict(raw)))

    def save_if_changed():
        if cache == raw:
            return
        _save_cache(cache, file_version + 1)
        print(f"\n💾 Cache v{file_version + 1} gespeichert unter '{CACHE_FILE}' ({len(cache)} Einträge).")

    pairs, skipped_by_filter = _collect_gguf_files(base_dirs)

    if skipped_by_filter:
        print(f"⏭️ {skipped_by_filter} Datei(en) durch Filter übersprungen ({', '.join(SKIP_PATH_SUBSTRINGS)}).")

    if not pairs:
        # Never prune here: nothing found usually means nothing mounted.
        print("❌ Keine GGUF-Dateien in den angegebenen Verzeichnissen gefunden.")
        save_if_changed()
        return cache

    errors = []
    skipped = []
    keyed, duplicates = _assign_keys(_build_candidates(pairs, errors))

    if duplicates:
        print(f"⏭️ {len(duplicates)} doppelte Datei(en) ignoriert:")
        for d in duplicates:
            print(f"  - {d}")

    stale = sorted(k for k in cache if k not in keyed)
    if stale and prune:
        for k in stale:
            del cache[k]
        print(f"🗑️ {len(stale)} Einträge ohne zugehörige Datei entfernt:")
        for k in stale:
            print(f"  - {k}")
    elif stale:
        print(f"ℹ️ {len(stale)} Cache-Einträge ohne gefundene Datei (entfernen mit --prune).")

    new_items = [(key, c) for key, c in keyed.items() if needs_scan(key, c.size, cache)]

    if not new_items:
        print("✅ Alles aktuell. Keine neuen oder geänderten GGUF-Dateien gefunden.")
    else:
        print(f"🔍 {len(new_items)} Modelle werden analysiert...")
        if os.path.exists(METADATA_DUMP_FILE):
            open(METADATA_DUMP_FILE, 'w').close()

        for key, c in tqdm(new_items, desc="GGUF Scan", unit="file", colour="green"):
            try:
                params = get_model_params(c.path, file_size_bytes=c.size)
                if c.num_shards:
                    params["num_shards"] = c.num_shards
                params["rel_path"] = key
                cache[key] = params
            except NotAnLLMError as e:
                skipped.append(f"Datei: {key} | Grund: {e}")
            except Exception as e:  # noqa: BLE001 -- one bad/corrupt GGUF file must not abort the whole batch scan
                errors.append(f"Datei: {key} | Grund: {e}")

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

    save_if_changed()
    return cache


def main():
    parser = argparse.ArgumentParser(description="GGUF-Dateien scannen und models_cache.json aktualisieren.")
    parser.add_argument("paths", nargs="*", help="Zu scannende Pfade (Default: bekannte Modell-Verzeichnisse)")
    parser.add_argument(
        "--prune", action="store_true",
        help="Cache-Einträge ohne gefundene Datei entfernen (nur wenn alle Modell-Verzeichnisse erreichbar sind)",
    )
    args = parser.parse_args()
    update_cache(args.paths or DEFAULT_DIRS, prune=args.prune)


if __name__ == "__main__":
    main()
