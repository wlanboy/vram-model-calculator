"""Einstiegspunkt für `uv run main.py <befehl>`, identisch mit dem installierten
Kommando `vram-model-calculator` (siehe vram_model_calculator/cli.py)."""
import sys

from vram_model_calculator.cli import main

if __name__ == "__main__":
    sys.exit(main(prog="main.py"))
