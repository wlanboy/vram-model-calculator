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
CACHE_VERSION = 2

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

# Newer types not yet in older gguf library versions.
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
    # Vendor-Prefix entfernen (z.B. "mistralai_", "Qwen_", "LiquidAI_", "allenai_")
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


def get_safe_int(reader, key):
    field = reader.fields.get(key)
    if not field:
        return None
    try:
        val = field.parts[-1]
        if hasattr(val, 'tolist'):
            val = val.tolist()
        if isinstance(val, list):
            val = val[0]
        return int(val)
    except (TypeError, ValueError, IndexError):
        return None


def get_vocab_size(reader, arch):
    """Tries arch.vocab_size first, then counts tokenizer.ggml.tokens as fallback."""
    v = get_safe_int(reader, f"{arch}.vocab_size")
    if v:
        return v
    # tokenizer.ggml.tokens is a string array; field.data has one entry per token.
    field = reader.fields.get("tokenizer.ggml.tokens")
    if not field:
        return None
    try:
        return len(field.data)
    except Exception:
        return None


def get_mmproj_params(reader, file_size_bytes):
    return {
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


def get_model_params(file_path):
    reader = GGUFReader(file_path)
    file_size_bytes = os.path.getsize(file_path)

    if os.path.basename(file_path).startswith("mmproj-"):
        return get_mmproj_params(reader, file_size_bytes)

    arch = get_str(reader, "general.architecture")
    if not arch:
        print(f"  ⚠️ Keine Architektur-Metadaten in {os.path.basename(file_path)}, nutze 'llama' als Fallback.")
        arch = "llama"

    n_ctx = (get_safe_int(reader, f"{arch}.context_length") or
             get_safe_int(reader, "general.context_length") or 32768)

    file_type_id = get_safe_int(reader, "general.file_type")
    quant = FILE_TYPE_MAP.get(file_type_id, f"unknown({file_type_id})") if file_type_id is not None else None

    params = {
        "type": "llm",
        "arch": arch,
        "name": clean_name(get_str(reader, "general.name")),
        "quant": quant,
        "n_layers": get_safe_int(reader, f"{arch}.block_count"),
        "n_embd": get_safe_int(reader, f"{arch}.embedding_length"),
        "n_heads": get_safe_int(reader, f"{arch}.attention.head_count"),
        "n_kv_heads": get_safe_int(reader, f"{arch}.attention.head_count_kv") or None,
        "n_ff": get_safe_int(reader, f"{arch}.feed_forward_length"),
        "n_experts": get_safe_int(reader, f"{arch}.expert_count"),
        "n_experts_used": get_safe_int(reader, f"{arch}.expert_used_count"),
        "vocab_size": get_vocab_size(reader, arch),
        "n_ctx_orig": n_ctx,
        "file_size_bytes": file_size_bytes,
        "file_size_gb": round(file_size_bytes / (1024**3), 3),
    }

    if not params["n_layers"] or params["n_layers"] < 1:
        raise ValueError("Ungültige Metadaten (kein LLM?)")

    return params


def needs_scan(path, base_dir, cache):
    rel = os.path.relpath(path, base_dir)
    if rel not in cache:
        return True
    return cache[rel].get("file_size_bytes") != os.path.getsize(path)


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

    new_files = [f for f in all_files if needs_scan(f, base_dir, cache)]

    if not new_files:
        print("✅ Alles aktuell. Keine neuen oder geänderten GGUF-Dateien gefunden.")
        return cache

    print(f"🔍 {len(new_files)} Modelle werden analysiert...")

    for path in tqdm(new_files, desc="GGUF Scan", unit="file", colour="green"):
        rel = os.path.relpath(path, base_dir)
        try:
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

    print(f"\n💾 Cache gespeichert unter '{CACHE_FILE}' ({len(cache)} Einträge).")
    return cache


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    update_cache(target)
