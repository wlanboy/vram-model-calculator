"""Low-level GGUF field readers.

Pulls typed values out of a GGUFReader without knowing anything about model
architectures, naming conventions, or capability detection.
"""
from gguf import GGUFReader

try:
    from gguf.constants import LlamaFileType
    FILE_TYPE_MAP = {
        e.value: e.name.replace("MOSTLY_", "").replace("ALL_", "")
        for e in LlamaFileType
    }
except ImportError:
    # Mirrors gguf-py's LlamaFileType enum (checked against llama.cpp
    # gguf-py/gguf/constants.py, commit 6a1a922d, 2026-09). IDs 4-6 and
    # 33-35 are retired/reserved (old Q4_1_SOME_F16/Q4_2/Q4_3 and the
    # Q4_0_4_4/4_8/8_8 repack formats, respectively) and left unmapped.
    FILE_TYPE_MAP = {
        0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0",
        8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M",
        13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M",
        18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS",
        23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
        28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
        36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4_MOE", 39: "NVFP4",
        40: "Q1_0", 41: "Q2_0",
    }

try:
    from gguf.constants import GGUFValueType as _GVT
    _STRING_TYPE = _GVT.STRING
except ImportError:
    _STRING_TYPE = None


def open_gguf_reader(file_path):
    try:
        return GGUFReader(file_path)
    except (ValueError, Exception) as e:
        msg = str(e)
        if "reshape" not in msg and "GGMLQuantizationType" not in msg:
            raise
        # Tensor data loading failed (unsupported quant layout, e.g. a tensor
        # dtype ID the installed gguf lib doesn't know yet) — retry reading
        # metadata-only by temporarily suppressing _build_tensors.
        original = GGUFReader._build_tensors
        GGUFReader._build_tensors = lambda self, *a, **kw: None
        try:
            return GGUFReader(file_path)
        finally:
            GGUFReader._build_tensors = original


def field_is_string(field):
    try:
        return field.types[0] == _STRING_TYPE
    except (AttributeError, IndexError):
        return False


def decode_bytes(val, errors='strict'):
    """Decodes a GGUF byte-array field value as UTF-8, stripping NUL padding."""
    return val.tobytes().decode('utf-8', errors=errors).strip('\x00')


def iter_decoded_parts(field, errors='replace'):
    """Yields each part of a GGUF field, decoded as a lowercase string."""
    for part in field.parts:
        try:
            yield decode_bytes(part, errors=errors).lower()
        except (AttributeError, IndexError, UnicodeDecodeError):
            continue


def get_str(reader, key):
    field = reader.fields.get(key)
    if not field:
        return None
    try:
        val = field.parts[-1]
        if hasattr(val, 'tobytes'):
            return decode_bytes(val)
        return str(val)
    except (AttributeError, IndexError, UnicodeDecodeError):
        return None


def _get_int(reader, keys, accept):
    for key in keys:
        field = reader.fields.get(key)
        if not field:
            continue
        try:
            val = field.parts[-1]
            if hasattr(val, 'tolist'):
                val = val.tolist()
            if isinstance(val, list):
                val = val[0]
            result = int(val)
        except (TypeError, ValueError, IndexError):
            continue
        if accept(result):
            return result
    return None


def get_safe_int(reader, *keys):
    """Try multiple keys in order, return first positive integer found."""
    return _get_int(reader, keys, lambda v: v > 0)


def get_nonneg_int(reader, *keys):
    """Try multiple keys in order, return first parsable integer found (0 is valid)."""
    return _get_int(reader, keys, lambda v: True)


def get_vocab_size(reader, arch):
    v = get_safe_int(reader, f"{arch}.vocab_size", "tokenizer.ggml.vocab_size")
    if v:
        return v
    field = reader.fields.get("tokenizer.ggml.tokens")
    if not field:
        return None
    try:
        return len(field.data)
    except (AttributeError, TypeError):
        return None
