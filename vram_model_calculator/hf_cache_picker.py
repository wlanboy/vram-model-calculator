"""Listet GGUF-Dateien im Huggingface-Cache auf und kopiert Auswahl nach /wdblack/models.

Der HF-Cache legt Snapshots unter
``~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/<hash>/<datei>.gguf``
ab. Dieses Tool nummeriert alle gefundenen GGUF-Dateien durch, damit man
gezielt einzelne (oder alle) davon nach ``/wdblack/models/<org>/<repo>/``
kopieren kann -- passend zur bestehenden Ordnerstruktur dort.
"""
import os
import re
import shutil
import sys

HF_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub/")
DEFAULT_DEST = "/wdblack/models"

REPO_DIR_RE = re.compile(r'^models--(?P<org>[^-].*?)--(?P<repo>.+)$')


def _human_size(num_bytes):
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def _split_repo_dir(repo_dir_name):
    """Zerlegt 'models--org--repo-name' in (org, repo-name).

    Der Repo-Name selbst darf '--' enthalten (wird nicht erneut gesplittet),
    nur der erste Teil nach 'models--' gilt als Org.
    """
    m = REPO_DIR_RE.match(repo_dir_name)
    if not m:
        return None, repo_dir_name
    return m.group("org"), m.group("repo")


def find_gguf_files(cache_dir=HF_CACHE_DIR):
    """Findet alle GGUF-Dateien im HF-Cache.

    Gibt eine Liste von dicts zurueck mit: path, org, repo, filename, size.
    Symlink-Ziele (blob-Dateien) werden ueber den Snapshot-Pfad aufgeloest.
    """
    results = []
    if not os.path.isdir(cache_dir):
        return results

    for entry in sorted(os.listdir(cache_dir)):
        repo_path = os.path.join(cache_dir, entry)
        if not os.path.isdir(repo_path) or not entry.startswith("models--"):
            continue
        org, repo = _split_repo_dir(entry)
        snapshots_dir = os.path.join(repo_path, "snapshots")
        if not os.path.isdir(snapshots_dir):
            continue
        seen_files = set()
        for snapshot in sorted(os.listdir(snapshots_dir)):
            snapshot_path = os.path.join(snapshots_dir, snapshot)
            if not os.path.isdir(snapshot_path):
                continue
            for fname in sorted(os.listdir(snapshot_path)):
                if not fname.lower().endswith(".gguf"):
                    continue
                if fname in seen_files:
                    continue
                fpath = os.path.join(snapshot_path, fname)
                if not os.path.isfile(fpath):
                    continue
                seen_files.add(fname)
                results.append({
                    "path": fpath,
                    "org": org or "unknown",
                    "repo": repo,
                    "filename": fname,
                    "size": os.path.getsize(fpath),
                })
    return results


def _parse_selection(text, max_index):
    """Parst '1,3,5-7' oder 'all' zu einer sortierten Liste von 1-basierten Indizes."""
    text = text.strip().lower()
    if text in ("all", "alle", "*"):
        return list(range(1, max_index + 1))

    indices = set()
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            start, _, end = part.partition("-")
            if start.isdigit() and end.isdigit():
                indices.update(range(int(start), int(end) + 1))
        elif part.isdigit():
            indices.add(int(part))
    return sorted(i for i in indices if 1 <= i <= max_index)


def _print_table(files):
    print(f"\nGefundene GGUF-Dateien in {HF_CACHE_DIR}:\n")
    for i, f in enumerate(files, start=1):
        print(f"[{i:3d}] {f['org']}/{f['repo']}/{f['filename']}  ({_human_size(f['size'])})")
    print()


def copy_selected(files, indices, dest_root=DEFAULT_DEST, dry_run=False):
    """Kopiert die ausgewaehlten Dateien nach dest_root/<org>/<repo>/<datei>."""
    copied, skipped = 0, 0
    for i in indices:
        f = files[i - 1]
        target_dir = os.path.join(dest_root, f["org"], f["repo"])
        target_path = os.path.join(target_dir, f["filename"])

        if os.path.exists(target_path) and os.path.getsize(target_path) == f["size"]:
            print(f"⏭️  {f['filename']} existiert bereits, übersprungen.")
            skipped += 1
            continue

        print(f"📦 {f['org']}/{f['repo']}/{f['filename']} ({_human_size(f['size'])}) -> {target_path}")
        if not dry_run:
            os.makedirs(target_dir, exist_ok=True)
            shutil.copy2(f["path"], target_path)
        copied += 1
    return copied, skipped


def main():
    raw_args = sys.argv[1:]
    dry_run = "--dry-run" in raw_args
    raw_args = [a for a in raw_args if a != "--dry-run"]

    dest_root = DEFAULT_DEST
    args = []
    i = 0
    while i < len(raw_args):
        if raw_args[i] == "--dest" and i + 1 < len(raw_args):
            dest_root = raw_args[i + 1]
            i += 2
            continue
        args.append(raw_args[i])
        i += 1

    files = find_gguf_files()
    if not files:
        print(f"❌ Keine GGUF-Dateien in {HF_CACHE_DIR} gefunden.")
        return 1

    _print_table(files)

    if args:
        selection = " ".join(args)
    else:
        selection = input("Auswahl (z.B. '1,3,5-7' oder 'all', leer = abbrechen): ")

    indices = _parse_selection(selection, len(files))
    if not indices:
        print("Keine gültige Auswahl, nichts kopiert.")
        return 0

    copied, skipped = copy_selected(files, indices, dest_root=dest_root, dry_run=dry_run)
    suffix = " (dry-run)" if dry_run else ""
    print(f"\n✅ {copied} Datei(en) kopiert, {skipped} übersprungen{suffix}. Ziel: {dest_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
