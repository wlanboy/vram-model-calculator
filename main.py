"""Einstiegspunkt fuer `uv run main.py`: buendelt scan/check/calculate."""
import argparse
import sys

from vram_model_calculator.gguf_checker import main as checker_main
from vram_model_calculator.gguf_scanner import DEFAULT_DIRS, update_cache
from vram_model_calculator.hf_cache_picker import DEFAULT_DEST
from vram_model_calculator.hf_cache_picker import main as hf_cache_picker_main
from vram_model_calculator.vram_calculator import calculate_vram_matrix


def main():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="GGUF Scanner, Checker und VRAM-Calculator fuer lokale LLMs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="Modelle scannen und models_cache.json aktualisieren")
    scan_parser.add_argument("paths", nargs="*", help="Zu scannende Pfade (Default: bekannte Modell-Verzeichnisse)")

    check_parser = subparsers.add_parser("check", help="GGUF-Dateien auf Integritaet pruefen")
    check_parser.add_argument("paths", nargs="*", help="Zu pruefende Pfade (Default: bekannte Modell-Verzeichnisse)")

    subparsers.add_parser("calculate", help="VRAM-Matrix aus models_cache.json berechnen und ausgeben")

    hf_parser = subparsers.add_parser(
        "hf-copy",
        help="GGUF-Dateien aus dem HF-Cache nummeriert auflisten und ausgewählte nach /wdblack/models kopieren",
    )
    hf_parser.add_argument("--dry-run", action="store_true", help="Nur anzeigen, nicht kopieren")
    hf_parser.add_argument("--dest", default=None, help=f"Zielverzeichnis (Default: {DEFAULT_DEST})")
    hf_parser.add_argument(
        "selection", nargs="*",
        help="Auswahl ('1,3,5-7' oder 'all'). Ohne Angabe wird interaktiv gefragt.",
    )

    args = parser.parse_args()

    if args.command == "scan":
        update_cache(args.paths or DEFAULT_DIRS)
        return 0
    if args.command == "check":
        sys.argv = [sys.argv[0], *args.paths]
        return checker_main()
    if args.command == "calculate":
        calculate_vram_matrix()
        return 0
    if args.command == "hf-copy":
        sys.argv = [
            sys.argv[0],
            *(["--dry-run"] if args.dry_run else []),
            *(["--dest", args.dest] if args.dest else []),
            *args.selection,
        ]
        return hf_cache_picker_main()
    return 1


if __name__ == "__main__":
    sys.exit(main())
