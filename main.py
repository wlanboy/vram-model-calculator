"""Einstiegspunkt fuer `uv run main.py`: buendelt scan/check/calculate."""
import argparse
import sys

from vram_model_calculator.gguf_checker import main as checker_main
from vram_model_calculator.gguf_scanner import DEFAULT_DIRS, update_cache
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
    return 1


if __name__ == "__main__":
    sys.exit(main())
