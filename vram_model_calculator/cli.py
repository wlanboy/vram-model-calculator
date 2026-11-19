"""Single entry point `vram-model-calculator <command>`. The old commands
gguf-scanner, gguf-checker, vram-calculator and hf-copy remain as aliases
for the matching subcommand."""
import argparse
import sys

from . import (
    families,
    gguf_checker,
    gguf_scanner,
    hardware,
    hf_cache_picker,
    vram_calculator,
)

COMMANDS = {
    "scan": (gguf_scanner, "Modelle scannen und models_cache.json aktualisieren"),
    "check": (gguf_checker, "GGUF-Dateien auf Integrität prüfen"),
    "calculate": (vram_calculator, "VRAM-Bedarf aus models_cache.json berechnen und ausgeben"),
    "hf-copy": (hf_cache_picker, "GGUF-Dateien aus dem HF-Cache auflisten und ins Modellverzeichnis kopieren"),
    "hardware": (hardware, "Tabellen der Hardware-Seiten aus device.json und dem Cache erzeugen"),
    "families": (families, "Modellfamilien-Navigation in index.html erzeugen"),
}


def build_parser(prog=None):
    parser = argparse.ArgumentParser(
        prog=prog,
        description="GGUF Scanner, Checker und VRAM-Calculator für lokale LLMs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, (module, help_text) in COMMANDS.items():
        sub = subparsers.add_parser(name, help=help_text, description=help_text)
        module.add_arguments(sub)
        sub.set_defaults(run=module.run)
    return parser


def main(argv=None, prog=None):
    args = build_parser(prog).parse_args(argv)
    try:
        return args.run(args)
    except ValueError as e:
        # Invalid config file and similar user errors: message, no traceback.
        print(f"❌ {e}")
        return 2


def _alias(command):
    """Console script for an old command name, e.g. gguf-scanner = scan."""
    def entry():
        return main([command, *sys.argv[1:]])
    return entry


scan = _alias("scan")
check = _alias("check")
calculate = _alias("calculate")
hf_copy = _alias("hf-copy")


if __name__ == "__main__":
    raise SystemExit(main())
