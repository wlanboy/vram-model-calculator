"""Heuristics for deriving a clean, human-readable model name from GGUF
metadata, with a fallback based on the file path when the metadata name
looks unreliable.
"""
import os
import re

_HEX_RE = re.compile(r'^[0-9a-fA-F]+$')
# Control characters and angle brackets have no place in a model name. The
# name comes from third-party GGUF metadata or file paths and ends up in the
# HTML of the public page, so drop them at the source as a second line of
# defence behind the escaping in filter.js.
_UNSAFE_CHARS_RE = re.compile(r'[\x00-\x1f\x7f<>]')


def strip_unsafe_chars(name):
    return _UNSAFE_CHARS_RE.sub('', name) if name else name


def clean_name(name):
    name = strip_unsafe_chars(name)
    if not name:
        return name
    # Strip HuggingFace org/user prefix: "allenai_olmOCR", "Ibm Granite_Granite 4.0", "Zai org_GLM"
    if '_' in name:
        prefix, rest = name.split('_', 1)
        prefix_letters = prefix.replace(' ', '')
        if rest and prefix_letters.isalpha() and 2 <= len(prefix_letters) <= 25:
            name = rest
    # Strip format suffixes (space, dash, or underscore as separator)
    name = re.sub(r'(\s+|[-_])(GGUF|AWQ|GPTQ|EXL2|MLX)$', '', name, flags=re.IGNORECASE)
    # Strip trailing quant/precision labels
    name = re.sub(r'\s+(BF16|F16|F32|IQ\d+[_A-Z0-9]*|Q\d+[_K0-9A-Z]*)$', '', name, flags=re.IGNORECASE)
    return name.strip()


def is_unreliable_name(name):
    if not name:
        return True
    if len(name) <= 2:
        return True
    return bool(_HEX_RE.match(name) and len(name) >= 16)


def name_from_path(file_path):
    parts = os.path.normpath(file_path).split(os.sep)
    candidate = parts[-2] if len(parts) >= 2 else os.path.splitext(parts[-1])[0]
    candidate = re.sub(r'[-_]GGUF$', '', candidate, flags=re.IGNORECASE)
    candidate = re.sub(r'[-_](Q\d+|IQ\d+|F16|F32|BF16).*$', '', candidate, flags=re.IGNORECASE)
    return strip_unsafe_chars(candidate) or None


def resolve_name(raw_name, file_path):
    """Prefers the cleaned metadata name; falls back to a path-derived name
    when the metadata name is missing, too short, or a bare hex hash.
    """
    return name_from_path(file_path) if is_unreliable_name(raw_name) else raw_name
