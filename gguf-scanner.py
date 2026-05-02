# /// script
# dependencies = ["gguf", "tqdm"]
# ///

import os
import re
import json
import sys
from tqdm import tqdm
from gguf import GGUFReader

CACHE_FILE = "models_cache.json"
METADATA_DUMP_FILE = "model-metadata.txt"
CACHE_VERSION = 2

SHARD_RE = re.compile(r'-(\d{5})-of-(\d{5})\.gguf$', re.IGNORECASE)

# Pure SSM architectures with no attention layers (n_kv_heads not applicable)
SSM_ARCHS = {"mamba", "mamba2", "rwkv", "rwkv6", "rwkv7"}

# Build FILE_TYPE_MAP from the gguf library so newer quant types are included automatically.
try:
    from gguf.constants import LlamaFileType
    FILE_TYPE_MAP = {
        e.value: e.name.replace("MOSTLY_", "").replace("ALL_", "")
        for e in LlamaFileType
    }
except Exception:
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


def clean_name(name):
    """Bereinigt general.name: entfernt Vendor-Prefix, -GGUF-Suffix und Quant-Artefakte."""
    if not name:
        return name
    if '_' in name:
        name = name.split('_', 1)[1]
    # -GGUF-Suffix entfernen
    name = re.sub(r'[-_]GGUF$', '', name, flags=re.IGNORECASE)
    # Quant-Bezeichnungen im Namen entfernen (z.B. " BF16", " F16", " Q4_K_M")
    name = re.sub(r'\s+(BF16|F16|F32|Q\d+[_K0-9A-Z]*)$', '', name)
    return name.strip()


def get_str(reader, key):
    field = reader.fields.get(key)
    if not field:
        return None
    try:
        val = field.parts[-1]
        if hasattr(val, 'tobytes'):
            return val.tobytes().decode('utf-8').strip('\x00')
        return str(val)
    except Exception:
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
    except Exception:
        return None


try:
    from gguf.constants import GGUFValueType as _GVT
    _STRING_TYPE = _GVT.STRING
except Exception:
    _STRING_TYPE = None


def _field_is_string(field):
    try:
        return field.types[0] == _STRING_TYPE
    except Exception:
        return False


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
            except Exception as e:
                f.write(f"  {key}: <read error: {e}>\n")


def get_shard_info(path):
    """Returns (shard_index, total_shards) or None if not a shard file."""
    m = SHARD_RE.search(os.path.basename(path))
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def get_mmproj_params(reader, file_path, file_size_bytes):
    params = {
        "type": "mmproj",
        "name": clean_name(get_str(reader, "general.name")),
        "image_size": get_safe_int(reader, "clip.vision.image_size"),
        "patch_size": get_safe_int(reader, "clip.vision.patch_size"),
        "n_embd": get_safe_int(reader, "clip.vision.embedding_length"),
        "n_ff": get_safe_int(reader, "clip.vision.feed_forward_length"),
        "n_layers": get_safe_int(reader, "clip.vision.block_count"),
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
    reader = GGUFReader(file_path)
    if file_size_bytes is None:
        file_size_bytes = os.path.getsize(file_path)

    general_type = get_str(reader, "general.type")

    if general_type == "adapter":
        return {
            "type": "adapter",
            "name": clean_name(get_str(reader, "general.name")),
            "file_size_bytes": file_size_bytes,
            "file_size_gb": round(file_size_bytes / (1024**3), 3),
        }

    if os.path.basename(file_path).startswith("mmproj-") or general_type == "projector":
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

    # Try multiple key variants per field for robustness
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

    # n_kv_heads: not applicable for pure SSM architectures
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

    params = {
        "type": "llm",
        "arch": arch,
        "name": clean_name(get_str(reader, "general.name")),
        "size_label": get_str(reader, "general.size_label"),
        "parameter_count": get_safe_int(reader, "general.parameter_count"),
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

    # Fields critical for VRAM calculation
    critical = ["n_layers", "n_embd", "vocab_size"]
    # n_kv_heads only critical for attention-based architectures
    if arch_lower not in SSM_ARCHS:
        critical.append("n_kv_heads")

    missing = [f for f in critical if params.get(f) is None]
    if missing:
        print(f"  ⚠️ Fehlende Felder {missing} in {os.path.basename(file_path)} → dump nach {METADATA_DUMP_FILE}")
        dump_all_fields(reader, file_path)
        params["has_missing_fields"] = True

    return params


def needs_scan(path, base_dir, cache):
    rel = os.path.relpath(path, base_dir)
    if rel not in cache:
        return True
    entry = cache[rel]
    if entry.get("file_size_bytes") != os.path.getsize(path):
        return True
    # Re-scan entries that previously had missing fields
    return entry.get("has_missing_fields", False)


def update_cache(base_dir):
    cache = {}
    errors = []

    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                loaded = json.load(f)
            if loaded.get("_version") == CACHE_VERSION:
                cache = {k: v for k, v in loaded.items() if k != "_version"}
            else:
                print("♻️ Cache-Version veraltet, wird neu aufgebaut...")
        except Exception as e:
            print(f"⚠️ Cache-Datei korrupt, erstelle neu. ({e})")

    if not os.path.exists(base_dir):
        print(f"❌ Pfad nicht gefunden: {base_dir}")
        return {}

    all_files = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".gguf"):
                all_files.append(os.path.join(root, f))

    # Group shard files: only process first shard, aggregate total size
    shard_groups = {}  # base_path -> sorted list of (shard_idx, path)
    non_shard_files = []
    for path in all_files:
        info = get_shard_info(path)
        if info:
            base = SHARD_RE.sub('.gguf', path)
            shard_groups.setdefault(base, []).append((info[0], path))
        else:
            non_shard_files.append(path)

    shard_meta = {}  # first_shard_path -> (total_size_bytes, num_shards)
    representative_files = list(non_shard_files)
    for base, shards in shard_groups.items():
        shards.sort(key=lambda x: x[0])
        first_path = shards[0][1]
        total_size = sum(os.path.getsize(p) for _, p in shards)
        shard_meta[first_path] = (total_size, len(shards))
        representative_files.append(first_path)

    new_files = [f for f in representative_files if needs_scan(f, base_dir, cache)]

    if not new_files:
        print("✅ Alles aktuell. Keine neuen oder geänderten GGUF-Dateien gefunden.")
        return cache

    print(f"🔍 {len(new_files)} Modelle werden analysiert...")

    # Clear the dump file for this run
    if os.path.exists(METADATA_DUMP_FILE):
        open(METADATA_DUMP_FILE, 'w').close()

    for path in tqdm(new_files, desc="GGUF Scan", unit="file", colour="green"):
        rel = os.path.relpath(path, base_dir)
        try:
            if path in shard_meta:
                total_size, num_shards = shard_meta[path]
                params = get_model_params(path, file_size_bytes=total_size)
                params["num_shards"] = num_shards
            else:
                params = get_model_params(path)
            params["rel_path"] = rel
            cache[rel] = params
        except Exception as e:
            errors.append(f"Datei: {rel} | Grund: {e}")

    with open(CACHE_FILE, 'w') as f:
        json.dump({"_version": CACHE_VERSION, **cache}, f, indent=4)

    if errors:
        print("\n⚠️ SCAN-FEHLER:")
        for err in errors:
            print(f"  - {err}")

    dump_count = sum(1 for v in cache.values() if isinstance(v, dict) and v.get("has_missing_fields"))
    if dump_count:
        print(f"\n📄 {dump_count} Modell(e) mit fehlenden Feldern → Details in '{METADATA_DUMP_FILE}'")

    print(f"\n💾 Cache gespeichert unter '{CACHE_FILE}' ({len(cache)} Einträge).")
    return cache


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    update_cache(target)
