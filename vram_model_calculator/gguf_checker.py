import os
import struct
import sys

from gguf import GGUFReader
from tqdm import tqdm

from .gguf_scanner import DEFAULT_DIRS, SHARD_RE, get_shard_info

GGUF_MAGIC = b'GGUF'
SUPPORTED_VERSIONS = (2, 3)


def _check_header(file_path):
    """Reads the raw magic bytes and version directly, independent of GGUFReader."""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(8)
    except OSError as e:
        return f"Datei nicht lesbar: {e}"

    if len(header) < 8:
        return f"Datei zu kurz für GGUF-Header ({len(header)} von 8 Bytes)"
    if header[:4] != GGUF_MAGIC:
        return f"Ungültige Magic-Bytes: {header[:4]!r} (erwartet {GGUF_MAGIC!r})"
    version = struct.unpack('<I', header[4:8])[0]
    if version not in SUPPORTED_VERSIONS:
        return f"Nicht unterstützte GGUF-Version: {version}"
    return None


def check_file(file_path):
    """Checks a single GGUF file for structural correctness and completeness.

    Returns a dict with at least a "status" key: "ok", "incomplete" (file is
    truncated / missing tensor data) or "corrupt" (malformed header/metadata).
    """
    if not os.path.isfile(file_path):
        return {"status": "corrupt", "reason": "Datei nicht gefunden"}

    header_error = _check_header(file_path)
    if header_error:
        return {"status": "corrupt", "reason": header_error}

    actual_size = os.path.getsize(file_path)

    try:
        reader = GGUFReader(file_path)
    except Exception as e:  # noqa: BLE001 -- any parse failure must be classified, not propagated
        msg = str(e)
        # A truncated file typically breaks while reshaping the last readable
        # tensor's data, or runs out of bytes while parsing kv/tensor-info
        # sections (IndexError from numpy on an out-of-range memmap slice).
        if isinstance(e, IndexError) or "reshape" in msg.lower():
            return {"status": "incomplete", "reason": f"Datei bricht mitten in den Daten ab: {e}"}
        return {"status": "corrupt", "reason": msg}

    if not reader.tensors:
        return {"status": "corrupt", "reason": "Keine Tensoren im Header gefunden"}

    expected_size = reader.data_offset
    for t in reader.tensors:
        expected_size = max(expected_size, t.data_offset + t.n_bytes)

    if actual_size < expected_size:
        return {
            "status": "incomplete",
            "reason": f"{expected_size - actual_size} Bytes fehlen "
                      f"(erwartet mind. {expected_size}, vorhanden {actual_size})",
            "expected_bytes": expected_size,
            "actual_bytes": actual_size,
        }

    return {
        "status": "ok",
        "expected_bytes": expected_size,
        "actual_bytes": actual_size,
        "n_tensors": len(reader.tensors),
    }


def check_shard_group(paths):
    """Checks whether a set of split GGUF shards (name-NNNNN-of-MMMMM.gguf) is complete.

    Only checks that all shard indices 1..total are present among `paths`;
    per-file integrity of each shard is checked separately via check_file().
    """
    shards = [(info[0], info[1]) for p in paths if (info := get_shard_info(p))]
    if not shards:
        return {"status": "corrupt", "reason": "Keine Shard-Dateien übergeben"}

    total = shards[0][1]
    found = sorted(idx for idx, _ in shards)
    missing = sorted(set(range(1, total + 1)) - set(found))

    if missing:
        return {"status": "incomplete", "reason": f"Fehlende Shards: {missing} von {total}", "total_shards": total}
    return {"status": "ok", "total_shards": total, "found_shards": len(shards)}


def _collect_gguf_files(targets):
    files = []
    for target in targets:
        if os.path.isfile(target):
            if target.endswith(".gguf"):
                files.append(os.path.abspath(target))
            continue
        if not os.path.isdir(target):
            print(f"❌ Pfad nicht gefunden: {target}")
            continue
        for root, _, names in os.walk(target):
            for n in names:
                if n.endswith(".gguf"):
                    files.append(os.path.join(root, n))
    return files


class ScanReport:
    """Accumulates check_file()/check_shard_group() results, partitioned by status."""

    def __init__(self):
        self.ok = []
        self.incomplete = []
        self.corrupt = []

    def add(self, result, path):
        reason = result.get("reason", "")
        if result["status"] == "ok":
            self.ok.append(path)
        elif result["status"] == "incomplete":
            self.incomplete.append((path, reason))
        else:
            self.corrupt.append((path, reason))

    @property
    def has_problems(self):
        return bool(self.incomplete or self.corrupt)


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_DIRS
    files = _collect_gguf_files(targets)
    if not files:
        print("❌ Keine GGUF-Dateien gefunden.")
        return 1

    shard_groups = {}
    singles = []
    for f in files:
        info = get_shard_info(f)
        if info:
            base = SHARD_RE.sub('.gguf', f)
            shard_groups.setdefault(base, []).append(f)
        else:
            singles.append(f)

    report = ScanReport()

    for f in tqdm(singles, desc="Einzeldateien prüfen", unit="file", colour="green"):
        report.add(check_file(f), f)

    for base, paths in tqdm(shard_groups.items(), desc="Shard-Gruppen prüfen", unit="group", colour="cyan"):
        for p in paths:
            report.add(check_file(p), p)
        group_result = check_shard_group(paths)
        if group_result["status"] != "ok":
            report.incomplete.append((base, group_result.get("reason", "")))

    print(f"\n✅ OK: {len(report.ok)}")
    if report.incomplete:
        print(f"⚠️ Unvollständig: {len(report.incomplete)}")
        for path, reason in report.incomplete:
            print(f"  - {path}: {reason}")
    if report.corrupt:
        print(f"❌ Defekt: {len(report.corrupt)}")
        for path, reason in report.corrupt:
            print(f"  - {path}: {reason}")

    return 1 if report.has_problems else 0


if __name__ == "__main__":
    sys.exit(main())
