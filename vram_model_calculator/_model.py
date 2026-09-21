import os
from dataclasses import dataclass

from .detection import METADATA_DUMP_FILE, detect_mcp, detect_thinking, dump_all_fields
from .gguf_fields import (
    FILE_TYPE_MAP,
    get_nonneg_int,
    get_safe_int,
    get_sliding_window,
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
# Also includes text-diffusion LLMs (Dream, LLaDA, RND1): unlike autoregressive
# models they don't decode token-by-token with a growing KV cache, so the
# KV-cache VRAM estimate below doesn't apply to them either.
DIFFUSION_ARCHS = {
    "flux", "sd1", "sd2", "sd3", "sdxl", "sdxl_refiner", "chroma",
    "lumina2", "auraflow", "hidream", "hunyuan_video", "wan", "wan2",
    "ltxv", "cosmos", "qwen_image", "pixart", "kolors", "cascade",
    "playground", "dream", "llada", "llada-moe", "rnd1",
}

# Multi-head Latent Attention (MLA) architectures (DeepSeek-V2/V3-style):
# instead of a per-head KV cache, they cache a single shared compressed
# vector per token per layer, sized kv_lora_rank + rope.dimension_count
# (see llama.cpp src/models/deepseek2.cpp). The generic n_kv_heads * head_dim
# formula wildly overstates their KV-cache VRAM.
MLA_ARCHS = {"deepseek2", "deepseek2-ocr", "minicpm3", "glm-dsa", "mistral4"}

# Tensor-name prefixes unique to stable-diffusion.cpp checkpoint GGUFs.
# Some (e.g. dreamshaper-xl-v2-turbo-Q8_0.gguf) carry no metadata at all —
# not even general.architecture — so DIFFUSION_ARCHS can't catch them.
# These prefixes never appear in an autoregressive-LLM GGUF (which names
# tensors "blk.N....."), so they're a reliable fallback signal.
DIFFUSION_TENSOR_PREFIXES = (
    "model.diffusion_model.", "first_stage_model.", "conditioner.embedders.",
    "cond_stage_model.", "double_blocks.", "single_blocks.",
)


def _is_metadata_less_diffusion_checkpoint(reader):
    return any(
        t.name.startswith(DIFFUSION_TENSOR_PREFIXES)
        for t in getattr(reader, "tensors", [])
    )

# KV-cache dtype is fp16 (2 bytes/element); each layer stores one Key and one
# Value tensor.
KV_BYTES_PER_ELEMENT = 2
KV_TENSORS_PER_LAYER = 2


class NotAnLLMError(ValueError):
    """Raised when a GGUF file is recognized as a non-LLM model (e.g. image diffusion)."""


@dataclass(frozen=True)
class ModelShape:
    """The handful of GGUF dimensions that determine KV-cache size.

    These fields only ever mean anything together, so they travel as one
    value instead of loose parameters. `kv_bytes_per_ctx_token()` and
    `kv_bytes_per_ctx_token_swa()` are the single source of truth for the
    KV-cache formula: their results are stored in each cache entry as
    `kv_bytes_per_ctx_token`/`kv_bytes_per_ctx_token_swa`, and every consumer
    (vram_calculator.py, filter.js) computes
    `kv_bytes_per_ctx_token * ctx + kv_bytes_per_ctx_token_swa * min(ctx, swa_window)`
    instead of re-deriving the per-architecture math itself.
    """

    n_layers: int
    n_embd: int
    n_heads: int
    n_kv_heads: int | None  # None => SSM/hybrid: no classic per-head KV cache
    mla_kv_dim: int | None = None
    # Sliding-window attention (Gemma2/3/4, Cohere2, gpt-oss, ...): swa_layers
    # of the model's n_layers cache only swa_window tokens each, regardless of
    # context length, instead of scaling with the full context like the rest.
    swa_layers: int = 0
    swa_window: int | None = None

    @property
    def is_ssm(self):
        return self.n_kv_heads is None

    def _kv_dim_bytes_per_layer(self):
        if self.is_ssm or not self.n_kv_heads:
            return 0
        head_dim = self.n_embd // (self.n_heads or 1)
        kv_dim = self.n_kv_heads * head_dim
        return KV_TENSORS_PER_LAYER * kv_dim * KV_BYTES_PER_ELEMENT

    def kv_bytes_per_ctx_token(self):
        """Bytes of KV-cache per context token contributed by the model's
        full/global-attention layers (all layers, unless swa_layers carves
        some out into kv_bytes_per_ctx_token_swa instead)."""
        if self.mla_kv_dim:
            # MLA architectures (DeepSeek2, GLM-DSA, Mistral4, MiniCPM3) cache
            # a single shared compressed vector per token/layer instead of a
            # value per KV-head. They don't combine with sliding-window
            # attention in practice, so swa_layers is ignored here.
            return self.n_layers * self.mla_kv_dim * KV_BYTES_PER_ELEMENT
        # swa_layers only comes out of the global count once swa_window is
        # set too, so a caller that leaves swa_window unset (or 0) still gets
        # the correct total: every layer counted as full/global attention.
        swa_layers = self.swa_layers if self.swa_window else 0
        global_layers = self.n_layers - swa_layers
        return global_layers * self._kv_dim_bytes_per_layer()

    def kv_bytes_per_ctx_token_swa(self):
        """Bytes of KV-cache per context token contributed by the model's
        local/sliding-window-attention layers, to be multiplied by
        min(ctx, swa_window) rather than the raw context length."""
        if self.mla_kv_dim or not self.swa_window:
            return 0
        return self.swa_layers * self._kv_dim_bytes_per_layer()


def get_mmproj_params(reader, file_path, file_size_bytes):
    raw_name = clean_name(get_str(reader, "general.name"))
    # Audio-only projectors (e.g. speech-to-text encoders) carry clip.audio.*
    # metadata instead of clip.vision.*; clip.has_audio_encoder marks those,
    # and clip.vision.* is absent since there's no vision tower to describe.
    is_audio = (
        get_safe_int(reader, "clip.has_audio_encoder")
        and reader.fields.get("clip.vision.embedding_length") is None
    )
    if is_audio:
        params = {
            "type": MODEL_TYPE_MMPROJ,
            "name": resolve_name(raw_name, file_path),
            "modality": "audio",
            "num_mel_bins": get_safe_int(reader, "clip.audio.num_mel_bins"),
            "n_embd": get_safe_int(reader, "clip.audio.embedding_length"),
            "n_ff": get_safe_int(reader, "clip.audio.feed_forward_length"),
            "n_layers": get_nonneg_int(reader, "clip.audio.block_count"),
            "projection_dim": get_safe_int(reader, "clip.audio.projection_dim"),
            "has_llava_projector": get_safe_int(reader, "clip.has_llava_projector"),
            "file_size_bytes": file_size_bytes,
            "file_size_gb": round(file_size_bytes / (1024**3), 3),
        }
        critical = ["num_mel_bins", "n_embd", "n_layers"]
    else:
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
        if _is_metadata_less_diffusion_checkpoint(reader):
            raise NotAnLLMError(
                "kein LLM, Diffusionsmodell (keine general.architecture-Metadaten, "
                "SD-Tensornamen erkannt)"
            )
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

    mla_kv_dim = None
    if arch_lower in MLA_ARCHS:
        kv_lora_rank = get_safe_int(reader, f"{arch}.attention.kv_lora_rank")
        rope_dim = get_safe_int(reader, f"{arch}.rope.dimension_count")
        if kv_lora_rank and rope_dim:
            mla_kv_dim = kv_lora_rank + rope_dim

    swa_window, swa_layers = get_sliding_window(reader, arch, n_layers or 0)

    raw_name = clean_name(get_str(reader, "general.name"))
    name = resolve_name(raw_name, file_path)

    shape = ModelShape(
        n_layers=n_layers or 0,
        n_embd=n_embd or 0,
        n_heads=n_heads or 0,
        n_kv_heads=n_kv_heads,
        mla_kv_dim=mla_kv_dim,
        swa_layers=swa_layers,
        swa_window=swa_window,
    )

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
        "mla_kv_dim": mla_kv_dim,
        "kv_bytes_per_ctx_token": shape.kv_bytes_per_ctx_token(),
        "kv_bytes_per_ctx_token_swa": shape.kv_bytes_per_ctx_token_swa(),
        "swa_layers": swa_layers,
        "swa_window": swa_window,
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
