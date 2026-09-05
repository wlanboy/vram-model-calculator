import os

from .detection import METADATA_DUMP_FILE, detect_mcp, detect_thinking, dump_all_fields
from .gguf_fields import (
    FILE_TYPE_MAP,
    get_nonneg_int,
    get_safe_int,
    get_str,
    get_vocab_size,
    open_gguf_reader,
)
from .name_utils import clean_name, resolve_name

# Model kind, as stored in each cache entry's "type" field.
MODEL_TYPE_LLM = "llm"
MODEL_TYPE_ADAPTER = "adapter"
MODEL_TYPE_MMPROJ = "mmproj"

# Pure SSM and hybrid SSM/attention architectures where n_kv_heads is absent
# from the GGUF metadata (n_kv_heads not applicable, or not exposed as a
# single global value for the hybrid attention layers)
SSM_ARCHS = {
    "mamba", "mamba2", "rwkv", "rwkv6", "rwkv7", "rwkv6qwen2", "arwkv7",
    "jamba", "falcon-h1", "granitehybrid", "plamo2", "plamo3",
    "qwen3next", "lfm2", "lfm2moe", "nemotron_h", "nemotron_h_moe",
    "qwen35", "qwen35moe", "qwen4exp", "kimi-linear", "kimi-k3",
    "bailingmoe3", "minimax-01",
}

# Image/video diffusion architectures (stable-diffusion.cpp GGUF quantizations,
# e.g. from HF caches shared with LMStudio/HF hub). These carry no LLM-style
# block_count/n_layers metadata and are out of scope for this VRAM calculator.
DIFFUSION_ARCHS = {
    "flux", "sd1", "sd2", "sd3", "sdxl", "sdxl_refiner", "chroma",
    "lumina2", "auraflow", "hidream", "hunyuan_video", "wan", "wan2",
    "ltxv", "cosmos", "qwen_image", "pixart", "kolors", "cascade",
    "playground",
}


class NotAnLLMError(ValueError):
    """Raised when a GGUF file is recognized as a non-LLM model (e.g. image diffusion)."""


def get_mmproj_params(reader, file_path, file_size_bytes):
    raw_name = clean_name(get_str(reader, "general.name"))
    params = {
        "type": MODEL_TYPE_MMPROJ,
        "name": resolve_name(raw_name, file_path),
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
    reader = open_gguf_reader(file_path)
    if file_size_bytes is None:
        file_size_bytes = os.path.getsize(file_path)

    general_type = get_str(reader, "general.type")

    if general_type == "adapter":
        raw_name = clean_name(get_str(reader, "general.name"))
        return {
            "type": MODEL_TYPE_ADAPTER,
            "name": resolve_name(raw_name, file_path),
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

    if arch_lower in DIFFUSION_ARCHS:
        raise NotAnLLMError(f"kein LLM, Diffusionsmodell (arch={arch})")

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
    name = resolve_name(raw_name, file_path)

    params = {
        "type": MODEL_TYPE_LLM,
        "arch": arch,
        "name": name,
        "size_label": get_str(reader, "general.size_label"),
        "parameter_count": get_safe_int(reader, "general.parameter_count"),
        "mcp":      detect_mcp(reader, file_path),
        "thinking": detect_thinking(reader, name, file_path),
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
