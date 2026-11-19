"""Finding GGUF files and grouping split models (name-00001-of-00003.gguf),
shared by the scanner and the checker."""
import os
import re

SHARD_RE = re.compile(r'-(\d{5})-of-(\d{5})\.gguf$', re.IGNORECASE)


def is_gguf(name):
    return name.lower().endswith(".gguf")


def get_shard_info(path):
    """Returns (shard_index, total_shards) or None if not a shard file."""
    m = SHARD_RE.search(os.path.basename(path))
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def shard_group_name(path):
    """Common name of all shards of a split model (the path without -NNNNN-of-MMMMM)."""
    return SHARD_RE.sub('.gguf', path)


def collect_gguf_files(targets, skip_patterns=(), verbose=False):
    """Collects GGUF files from files and directories (recursively).

    Returns ([(abs_path, base_dir)], skipped_by_filter_count). base_dir is the
    directory target the file was found under (its own directory for a file
    target). skip_patterns are case-insensitive substrings matched against the
    path below base_dir only, so a base dir like /data/dolphin/ doesn't
    exclude everything inside it.
    """
    skip = [s.lower() for s in skip_patterns]
    pairs = []
    skipped = 0

    def add(abs_path, base_dir):
        nonlocal skipped
        rel_path = os.path.relpath(abs_path, base_dir).lower()
        if any(s in rel_path for s in skip):
            skipped += 1
        else:
            pairs.append((abs_path, base_dir))

    for target in targets:
        target = os.path.abspath(target)
        if os.path.isfile(target):
            if is_gguf(target):
                add(target, os.path.dirname(target))
            continue
        if not os.path.isdir(target):
            print(f"❌ Pfad nicht gefunden: {target}")
            continue
        if verbose:
            print(f"📂 Scanne: {target}")
        for root, _, names in os.walk(target):
            for n in names:
                if is_gguf(n):
                    add(os.path.join(root, n), target)
    return pairs, skipped


def group_shards(paths):
    """Splits paths into (singles, {group_name: [shard paths sorted by index]})."""
    singles = []
    groups = {}
    for p in paths:
        if get_shard_info(p):
            groups.setdefault(shard_group_name(p), []).append(p)
        else:
            singles.append(p)
    for shards in groups.values():
        shards.sort(key=lambda p: (get_shard_info(p) or (0, 0))[0])
    return singles, groups
