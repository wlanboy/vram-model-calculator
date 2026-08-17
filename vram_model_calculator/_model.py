import contextlib
import os
import re

from gguf import GGUFReader

METADATA_DUMP_FILE = "model-metadata.txt"


def _open_gguf_reader(file_path):
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

THINKING_NAME_RE = re.compile(
    r'(think(?:ing)?|qwq|deepseek[-_]?r\d+|reason(?:ing)?|logic|reflect|chain|cog)',
    re.IGNORECASE
)

# Pure SSM and hybrid SSM/attention architectures where n_kv_heads is absent
# from the GGUF metadata (n_kv_heads not applicable, or not exposed as a
# single global value for the hybrid attention layers)
SSM_ARCHS = {
    "mamba", "mamba2", "rwkv", "rwkv6", "rwkv7", "rwkv6qwen2", "arwkv7",
    "jamba", "falcon_h1", "granite_hybrid", "plamo2", "plamo3",
    "qwen3next", "lfm2", "lfm2moe", "nemotron_h", "nemotron_h_moe",
}

# Build FILE_TYPE_MAP from the gguf library so newer quant types are included automatically.
try:
    from gguf.constants import LlamaFileType
    FILE_TYPE_MAP = {
        e.value: e.name.replace("MOSTLY_", "").replace("ALL_", "")
        for e in LlamaFileType
    }
except ImportError:
    FILE_TYPE_MAP = {
        0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0",
        8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M",
        13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M",
        18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "IQ3_XXS", 22: "IQ1_S",
        23: "IQ4_NL", 24: "IQ3_S", 25: "IQ2_S", 26: "IQ4_XS", 27: "IQ1_M",
        28: "BF16",
    }

FILE_TYPE_MAP.setdefault(29, "Q4_0_4_4")
FILE_TYPE_MAP.setdefault(30, "Q4_0_4_8")
FILE_TYPE_MAP.setdefault(31, "Q4_0_8_8")
FILE_TYPE_MAP.setdefault(32, "TQ1_0")
FILE_TYPE_MAP.setdefault(33, "TQ2_0")
FILE_TYPE_MAP.setdefault(38, "MXFP4")

_HEX_RE = re.compile(r'^[0-9a-fA-F]+$')

# --- Name utilities ---

def clean_name(name):
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


def _is_unreliable_name(name):
    if not name:
        return True
    if len(name) <= 2:
        return True
    return bool(_HEX_RE.match(name) and len(name) >= 16)


def _name_from_path(file_path):
    parts = os.path.normpath(file_path).split(os.sep)
    candidate = parts[-2] if len(parts) >= 2 else os.path.splitext(parts[-1])[0]
    candidate = re.sub(r'[-_]GGUF$', '', candidate, flags=re.IGNORECASE)
    candidate = re.sub(r'[-_](Q\d+|IQ\d+|F16|F32|BF16).*$', '', candidate, flags=re.IGNORECASE)
    return candidate or None

# --- Low-level GGUF field readers ---

def get_str(reader, key):
    field = reader.fields.get(key)
    if not field:
        return None
    try:
        val = field.parts[-1]
        if hasattr(val, 'tobytes'):
            return val.tobytes().decode('utf-8').strip('\x00')
        return str(val)
    except (AttributeError, IndexError, UnicodeDecodeError):
        return None


def get_safe_int(reader, *keys):
    """Try multiple keys in order, return first positive integer found."""
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
            if result > 0:
                return result
        except (TypeError, ValueError, IndexError):
            continue
    return None


def get_nonneg_int(reader, *keys):
    """Try multiple keys in order, return first non-negative integer found (0 is valid)."""
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
            return int(val)
        except (TypeError, ValueError, IndexError):
            continue
    return None


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


try:
    from gguf.constants import GGUFValueType as _GVT
    _STRING_TYPE = _GVT.STRING
except ImportError:
    _STRING_TYPE = None


def _field_is_string(field):
    try:
        return field.types[0] == _STRING_TYPE
    except (AttributeError, IndexError):
        return False

# --- Debug dump ---

def _detect_mcp(reader, name, file_path):
    """Returns True if the model supports tool calls / MCP."""
    tmpl = get_str(reader, "tokenizer.chat_template")
    if tmpl:
        tl = tmpl.lower()
        if "tool_call" in tl or "function_call" in tl or "<|tool|>" in tl or "[tool_calls]" in tl:
            return True
    tags_field = reader.fields.get("general.tags")
    if tags_field:
        with contextlib.suppress(AttributeError, IndexError, UnicodeDecodeError):
            for part in tags_field.parts:
                tag = part.tobytes().decode("utf-8", errors="replace").strip("\x00").lower()
                if "tool" in tag or "function-call" in tag or "mcp" in tag:
                    return True
    return False


def _detect_thinking(reader, name, file_path):
    """Returns True if the model supports extended thinking/reasoning."""
    # Primary signal: chat template contains <think> token
    tmpl = get_str(reader, "tokenizer.chat_template")
    if tmpl and "<think>" in tmpl:
        return True
    # Secondary signal: general.tags contains "thinking" or "reasoning"
    tags_field = reader.fields.get("general.tags")
    if tags_field:
        with contextlib.suppress(AttributeError, IndexError, UnicodeDecodeError):
            for part in tags_field.parts:
                tag = part.tobytes().decode("utf-8", errors="replace").strip("\x00").lower()
                if "think" in tag or "reason" in tag:
                    return True
    # Fallback: name/path heuristic
    text = (name or "") + " " + os.path.basename(file_path)
    return bool(THINKING_NAME_RE.search(text))


def dump_all_fields(reader, file_path):
    """Appends all raw GGUF fields to METADATA_DUMP_FILE for debugging."""
    skip_keys = {"tokenizer.ggml.merges", "tokenizer.ggml.tokens", "tokenizer.ggml.token_type"}
    with open(METADATA_DUMP_FILE, 'a', encoding='utf-8') as f:
        f.write(f"\n{'=' * 80}\n")
        f.write(f"File: {file_path}\n")
        f.write(f"{'=' * 80}\n")
        for key in sorted(reader.fields.keys()):
            if key in skip_keys or key == "tokenizer.chat_template":
                continue
            field = reader.fields[key]
            try:
                val = field.parts[-1]
                if _field_is_string(field):
                    display = val.tobytes().decode('utf-8', errors='replace').strip('\x00')
                elif hasattr(val, 'tolist'):
                    lst = val.tolist()
                    display = lst[0] if isinstance(lst, list) and len(lst) == 1 else lst
                else:
                    display = str(val)
                f.write(f"  {key}: {display}\n")
            except (AttributeError, IndexError, TypeError, ValueError, UnicodeDecodeError) as e:
                f.write(f"  {key}: <read error: {e}>\n")

# --- Model parameter extraction ---

def get_mmproj_params(reader, file_path, file_size_bytes):
    raw_name = clean_name(get_str(reader, "general.name"))
    params = {
        "type": "mmproj",
        "name": _name_from_path(file_path) if _is_unreliable_name(raw_name) else raw_name,
        "image_size": get_safe_int(reader, "clip.vision.image_size"),
        "patch_size": get_safe_int(reader, "clip.vision.patch_size"),
        "n_embd": get_safe_int(reader, "clip.vision.embedding_length"),
        "n_ff": get_safe_int(reader, "clip.vision.feed_forward_length"),
        "n_layers": get_nonneg_int(reader, "clip.vision.block_count"),
        "projection_dim": get_safe_int(reader, "clip.vision.projection_dim"),
        "has_llava_projector": get_safe_int(reader, "clip.has_llava_projector"),
        "file_size_bytes": file_size_bytes,
        "file_size_gb": round(file_size_bytes / (1024**3), 3),
    }
    critical = ["image_size", "n_embd", "n_layers"]
    missing = [f for f in critical if params.get(f) is None]
    if missing:
        print(f"  ⚠️ Fehlende Felder {missing} in {os.path.basename(file_path)} → dump nach {METADATA_DUMP_FILE}")
        dump_all_fields(reader, file_path)
        params["has_missing_fields"] = True
    return params


def get_model_params(file_path, file_size_bytes=None):
    reader = _open_gguf_reader(file_path)
    if file_size_bytes is None:
        file_size_bytes = os.path.getsize(file_path)

    general_type = get_str(reader, "general.type")

    if general_type == "adapter":
        raw_name = clean_name(get_str(reader, "general.name"))
        return {
            "type": "adapter",
            "name": _name_from_path(file_path) if _is_unreliable_name(raw_name) else raw_name,
            "file_size_bytes": file_size_bytes,
            "file_size_gb": round(file_size_bytes / (1024**3), 3),
        }

    if "mmproj" in os.path.basename(file_path).lower() or general_type == "projector":
        return get_mmproj_params(reader, file_path, file_size_bytes)

    arch = get_str(reader, "general.architecture")
    if not arch:
        print(f"  ⚠️ Keine Architektur in {os.path.basename(file_path)}, nutze 'llama' als Fallback.")
        arch = "llama"

    arch_lower = arch.lower()

    n_ctx = (
        get_safe_int(reader, f"{arch}.context_length") or
        get_safe_int(reader, "general.context_length") or
        32768
    )

    file_type_id = get_safe_int(reader, "general.file_type")
    quant = FILE_TYPE_MAP.get(file_type_id, f"unknown({file_type_id})") if file_type_id is not None else None

    n_layers = get_safe_int(reader,
        f"{arch}.block_count",
        f"{arch}.num_hidden_layers",
        f"{arch}.layers",
    )
    n_embd = get_safe_int(reader,
        f"{arch}.embedding_length",
        f"{arch}.hidden_size",
        f"{arch}.d_model",
    )
    n_heads = get_safe_int(reader,
        f"{arch}.attention.head_count",
        f"{arch}.num_attention_heads",
        f"{arch}.attention.num_heads",
    )
    n_ff = get_safe_int(reader,
        f"{arch}.feed_forward_length",
        f"{arch}.intermediate_size",
        f"{arch}.ffn_hidden_size",
    )

    if arch_lower in SSM_ARCHS:
        n_kv_heads = None
    else:
        raw_kv = get_nonneg_int(reader,
            f"{arch}.attention.head_count_kv",
            f"{arch}.num_key_value_heads",
            f"{arch}.attention.kv_head_count",
        )
        # 0 means "same as n_heads" in llama.cpp convention
        n_kv_heads = n_heads if (raw_kv is not None and raw_kv == 0) else raw_kv

    raw_name = clean_name(get_str(reader, "general.name"))
    name = _name_from_path(file_path) if _is_unreliable_name(raw_name) else raw_name

    params = {
        "type": "llm",
        "arch": arch,
        "name": name,
        "size_label": get_str(reader, "general.size_label"),
        "parameter_count": get_safe_int(reader, "general.parameter_count"),
        "mcp":      _detect_mcp(reader, name, file_path),
        "thinking": _detect_thinking(reader, name, file_path),
        "quant": quant,
        "n_layers": n_layers,
        "n_embd": n_embd,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "n_ff": n_ff,
        "n_experts": get_safe_int(reader, f"{arch}.expert_count"),
        "n_experts_used": get_safe_int(reader, f"{arch}.expert_used_count"),
        "vocab_size": get_vocab_size(reader, arch),
        "n_ctx_orig": n_ctx,
        "file_size_bytes": file_size_bytes,
        "file_size_gb": round(file_size_bytes / (1024**3), 3),
    }

    if not params["n_layers"] or params["n_layers"] < 1:
        raise ValueError("Ungültige Metadaten (kein LLM?)")

    critical = ["n_layers", "n_embd", "vocab_size"]
    if arch_lower not in SSM_ARCHS:
        critical.append("n_kv_heads")

    missing = [f for f in critical if params.get(f) is None]
    if missing:
        print(f"  ⚠️ Fehlende Felder {missing} in {os.path.basename(file_path)} → dump nach {METADATA_DUMP_FILE}")
        dump_all_fields(reader, file_path)
        params["has_missing_fields"] = True

    return params
