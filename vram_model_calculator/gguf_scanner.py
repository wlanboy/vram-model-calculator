import argparse
import fnmatch
import json
import os
from collections import namedtuple

from tqdm import tqdm

from ._model import (
    MODEL_TYPE_LLM,
    SSM_ARCHS,
    ModelShape,
    NotAnLLMError,
    clean_name,
    compute_buffer_bytes,
    get_model_params,
)
from .config import (
    add_cache_argument,
    load_config,
    metadata_dump_file,
    resolve_cache_file,
)
from .files import collect_gguf_files, get_shard_info, group_shards

# One model to scan: a single file, or the first shard of a complete shard
# set (size is then the sum of all shards).
Candidate = namedtuple("Candidate", "path base_dir size num_shards")

# Integer fields of an LLM entry. Very old caches stored some of them as
# strings, partly as single characters holding the value as code point
# ('(' = 40); _migrate_legacy_ints converts those once.
INT_FIELDS = (
    "n_layers", "n_embd", "n_heads", "n_kv_heads", "n_ff", "n_ff_exp", "n_experts", "n_experts_used",
    "vocab_size", "n_ctx_orig", "mla_kv_dim", "head_dim_k", "head_dim_v", "head_dim_k_swa",
    "head_dim_v_swa", "kv_layers", "kv_bytes_per_ctx_token", "kv_bytes_per_ctx_token_swa",
    "swa_layers", "swa_window", "recurrent_layers", "recurrent_state_bytes", "compute_buffer_bytes",
    "file_size_bytes", "parameter_count",
)


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


def _legacy_int(val):
    """A single-character string is the value as code point, other strings
    are decimal; anything unparseable becomes None."""
    if len(val) == 1:
        return ord(val)
    try:
        return int(val)
    except ValueError:
        return None


def _migrate_legacy_ints(cache):
    """Converts string values in INT_FIELDS to int, so readers of the cache
    (vram_calculator.py, filter.js) only ever see ints or null."""
    migrated = 0
    for key, entry in cache.items():
        if not isinstance(entry, dict) or entry.get("type") != MODEL_TYPE_LLM:
            continue
        fixed = {f: _legacy_int(entry[f]) for f in INT_FIELDS if isinstance(entry.get(f), str)}
        if fixed:
            cache[key] = {**entry, **fixed}
            migrated += 1
    if migrated:
        print(f"🔧 {migrated} Cache-Einträge mit alten String-Werten migriert.")
    return cache


def _backfill_compute_buffer(cache):
    """Adds compute_buffer_bytes to entries scanned before that field
    existed, from vocab_size/n_embd/n_ff already in the cache. MoE entries
    lack n_ff_exp until their next rescan; the estimate then rests on the
    logits term, which dominates for current vocabularies anyway."""
    backfilled = 0
    for key, entry in cache.items():
        if not isinstance(entry, dict) or entry.get("type") != MODEL_TYPE_LLM:
            continue
        if "compute_buffer_bytes" in entry or not entry.get("n_embd"):
            continue
        cache[key] = {**entry, "compute_buffer_bytes": compute_buffer_bytes(
            entry.get("vocab_size"), entry["n_embd"], entry.get("n_ff"),
            entry.get("n_ff_exp"), entry.get("n_experts_used"),
        )}
        backfilled += 1
    if backfilled:
        print(f"🧮 {backfilled} Cache-Einträge um compute_buffer_bytes ergänzt.")
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


def matches_rescan(rel_key, patterns):
    """True if rel_key matches one of the --rescan patterns: shell-style
    globs, or case-insensitive substrings when a pattern has no wildcard."""
    key = rel_key.lower()
    for pattern in patterns:
        p = pattern.lower()
        if any(c in p for c in "*?["):
            if fnmatch.fnmatchcase(key, p):
                return True
        elif p in key:
            return True
    return False


def needs_scan(rel_key, file_size_bytes, cache, rescan_all=False, rescan_patterns=()):
    """file_size_bytes is the total size of the model (sum of all shards)."""
    if rescan_all or (rescan_patterns and matches_rescan(rel_key, rescan_patterns)):
        return True
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
    if entry.get("type") != MODEL_TYPE_LLM:
        return False
    if "head_dim_k" not in entry:
        return True
    # Same for SSM/hybrid entries scanned before the recurrent state was
    # computed: its dimensions (ssm.*, shortconv.*, wkv.*) aren't in the
    # cache, so it can't be backfilled without the GGUF file.
    return str(entry.get("arch", "")).lower() in SSM_ARCHS and "recurrent_state_bytes" not in entry


def _load_cache(cache_file):
    """Returns (raw_entries, version); empty on missing or corrupt file."""
    if not os.path.exists(cache_file):
        return {}, 0
    try:
        with open(cache_file, 'r') as f:
            loaded = json.load(f)
        version = loaded.get("_version", 0)
        return {k: v for k, v in loaded.items() if k != "_version"}, version
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError) as e:
        print(f"⚠️ Cache-Datei korrupt, erstelle neu. ({e})")
        return {}, 0


def _save_cache(cache, version, cache_file):
    """Writes atomically: an interrupted write must not destroy the existing cache."""
    tmp_file = cache_file + ".tmp"
    with open(tmp_file, 'w') as f:
        json.dump({"_version": version, **dict(sorted(cache.items()))}, f, indent=4)
    os.replace(tmp_file, cache_file)


def _build_candidates(pairs, errors):
    """Groups shard files into one candidate per complete shard set. Incomplete
    sets and unreadable files (e.g. broken symlinks) are reported in errors."""
    base_of = dict(pairs)
    singles, groups = group_shards([p for p, _ in pairs])
    candidates = []
    for path in singles:
        try:
            candidates.append(Candidate(path, base_of[path], os.path.getsize(path), None))
        except OSError as e:
            errors.append(f"Datei: {path} | Grund: {e}")

    for group, shards in groups.items():
        infos = [get_shard_info(p) or (0, 0) for p in shards]
        indices = [idx for idx, _ in infos]
        totals = {total for _, total in infos}
        expected = max(totals)
        if len(totals) != 1 or indices != list(range(1, expected + 1)):
            errors.append(
                f"Datei: {group} | Grund: unvollständiger Shard-Satz "
                f"(gefunden {indices}, erwartet 1-{expected})"
            )
            continue
        try:
            total_size = sum(os.path.getsize(p) for p in shards)
        except OSError as e:
            errors.append(f"Datei: {group} | Grund: {e}")
            continue
        candidates.append(Candidate(shards[0], base_of[shards[0]], total_size, expected))
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


def update_cache(base_dirs, prune=False, cache_file=None, skip_patterns=None,
                 rescan_all=False, rescan_patterns=()):
    """Scans base_dirs (directories or single files) and updates the cache.
    With prune=True, cache entries whose file wasn't found in this run are
    removed. Only use it when all model directories are reachable, an
    unmounted disk looks like deleted models. skip_patterns defaults to the
    configured patterns; pass [] to scan everything. rescan_all/
    rescan_patterns force a fresh scan of all or the matching entries."""
    if isinstance(base_dirs, str):
        base_dirs = [base_dirs]
    cache_file = cache_file or resolve_cache_file()
    if skip_patterns is None:
        skip_patterns = load_config().skip_patterns
    dump_file = metadata_dump_file(cache_file)

    raw, file_version = _load_cache(cache_file)
    cache = _backfill_compute_buffer(_backfill_kv_bytes(_migrate_legacy_ints(_refresh_names(dict(raw)))))

    def save_if_changed():
        if cache == raw:
            return
        _save_cache(cache, file_version + 1, cache_file)
        print(f"\n💾 Cache v{file_version + 1} gespeichert unter '{cache_file}' ({len(cache)} Einträge).")

    pairs, skipped_by_filter = collect_gguf_files(base_dirs, skip_patterns, verbose=True)

    if skipped_by_filter:
        print(f"⏭️ {skipped_by_filter} Datei(en) durch Filter übersprungen ({', '.join(skip_patterns)}).")

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

    new_items = [
        (key, c) for key, c in keyed.items()
        if needs_scan(key, c.size, cache, rescan_all, rescan_patterns)
    ]

    if not new_items:
        print("✅ Alles aktuell. Keine neuen oder geänderten GGUF-Dateien gefunden.")
    else:
        print(f"🔍 {len(new_items)} Modelle werden analysiert...")
        if os.path.exists(dump_file):
            open(dump_file, 'w').close()

        for key, c in tqdm(new_items, desc="GGUF Scan", unit="file", colour="green"):
            try:
                params = get_model_params(c.path, file_size_bytes=c.size, dump_file=dump_file)
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
        print(f"\n📄 {dump_count} Modell(e) mit fehlenden Feldern → Details in '{dump_file}'")

    save_if_changed()
    return cache


def add_arguments(parser):
    parser.add_argument("paths", nargs="*", help="Zu scannende Pfade (Default: scan_dirs aus der Konfiguration)")
    parser.add_argument(
        "--prune", action="store_true",
        help="Cache-Einträge ohne gefundene Datei entfernen (nur wenn alle Modell-Verzeichnisse erreichbar sind)",
    )
    parser.add_argument("--rescan-all", action="store_true", help="Alle gefundenen Modelle neu einlesen")
    parser.add_argument(
        "--rescan", action="append", default=[], metavar="PATTERN",
        help="Passende Cache-Keys neu einlesen (Teilstring oder Glob wie '*qwen*', mehrfach möglich)",
    )
    parser.add_argument("--no-skip", action="store_true", help="Namensfilter (skip_patterns) für diesen Lauf abschalten")
    add_cache_argument(parser)


def run(args):
    config = load_config()
    update_cache(
        args.paths or config.expanded_scan_dirs,
        prune=args.prune,
        cache_file=resolve_cache_file(args.cache, config),
        skip_patterns=[] if args.no_skip else config.skip_patterns,
        rescan_all=args.rescan_all,
        rescan_patterns=args.rescan,
    )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="GGUF-Dateien scannen und models_cache.json aktualisieren.")
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
