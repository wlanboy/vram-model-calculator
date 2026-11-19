"""Paths and user settings: cache file location and the optional config file
~/.config/vram-model-calculator/config.toml.

Lookup order for the cache file: --cache option, VRAM_CALC_CACHE, the
config file's cache_file, then models_cache.json in the current directory
(the GitHub Pages site expects it there).
"""
import os
import tomllib
from dataclasses import dataclass, field

CACHE_ENV = "VRAM_CALC_CACHE"
CONFIG_ENV = "VRAM_CALC_CONFIG"
DEFAULT_CACHE_FILE = "models_cache.json"
METADATA_DUMP_NAME = "model-metadata.txt"

# Well-known model locations, scanned automatically when no path is given on
# the command line. Missing entries (e.g. an unmounted /wdblack) are skipped
# without error. Override with scan_dirs in the config file.
DEFAULT_SCAN_DIRS = [
    "~/LMStudio/models/",
    "~/.lmstudio/models/",
    "~/.cache/huggingface/hub/",
    "~/models",
    "/models",
    "/wdblack/models",
]

# Target directory of hf-copy. Override with copy_dest in the config file.
DEFAULT_COPY_DEST = "/wdblack/models"

# Path substrings (case-insensitive) that exclude a model from scanning, e.g.
# abliterated/uncensored finetunes. Override with skip_patterns in the config
# file, or switch off for one run with `scan --no-skip`.
DEFAULT_SKIP_PATTERNS = ["heretic", "heresy", "uncensored", "abliterated", "huihui", "dolphin"]


class ConfigError(ValueError):
    """Invalid config file; cli.main reports it without a traceback."""


@dataclass(frozen=True)
class Config:
    scan_dirs: list[str] = field(default_factory=lambda: list(DEFAULT_SCAN_DIRS))
    copy_dest: str = DEFAULT_COPY_DEST
    skip_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_SKIP_PATTERNS))
    cache_file: str | None = None

    @property
    def expanded_scan_dirs(self):
        return [os.path.expanduser(d) for d in self.scan_dirs]


def config_file_path():
    explicit = os.environ.get(CONFIG_ENV)
    if explicit:
        return os.path.expanduser(explicit)
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, "vram-model-calculator", "config.toml")


def _str_list(data, key, default):
    value = data.get(key, default)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"'{key}' muss eine Liste von Strings sein")
    return list(value)


def _str(data, key, default):
    value = data.get(key, default)
    if not isinstance(value, str):
        raise ConfigError(f"'{key}' muss ein String sein")
    return value


def _opt_str(data, key, default):
    value = data.get(key, default)
    if value is not None and not isinstance(value, str):
        raise ConfigError(f"'{key}' muss ein String sein")
    return value


def load_config(path=None):
    """Reads the config file. A missing file yields the defaults; an invalid
    one raises ValueError naming the file and the problem."""
    path = path or config_file_path()
    if not os.path.isfile(path):
        return Config()
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return Config(
            scan_dirs=_str_list(data, "scan_dirs", DEFAULT_SCAN_DIRS),
            copy_dest=os.path.expanduser(_str(data, "copy_dest", DEFAULT_COPY_DEST)),
            skip_patterns=_str_list(data, "skip_patterns", DEFAULT_SKIP_PATTERNS),
            cache_file=_opt_str(data, "cache_file", None),
        )
    except (tomllib.TOMLDecodeError, ValueError) as e:
        raise ConfigError(f"Konfigurationsdatei {path} ungültig: {e}") from e


def resolve_cache_file(path=None, config=None):
    """Cache file to use: explicit path, VRAM_CALC_CACHE, config, default."""
    chosen = path or os.environ.get(CACHE_ENV) or (config.cache_file if config else None)
    return os.path.expanduser(chosen) if chosen else DEFAULT_CACHE_FILE


def metadata_dump_file(cache_file):
    """model-metadata.txt lives next to the cache file it belongs to."""
    return os.path.join(os.path.dirname(os.path.abspath(cache_file)), METADATA_DUMP_NAME)


def add_cache_argument(parser):
    parser.add_argument(
        "--cache", metavar="PATH", default=None,
        help=f"Pfad zu models_cache.json (Default: ${CACHE_ENV}, cache_file aus der Konfiguration "
             f"oder {DEFAULT_CACHE_FILE} im aktuellen Verzeichnis)",
    )
