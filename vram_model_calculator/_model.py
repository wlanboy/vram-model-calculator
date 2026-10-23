import os
from dataclasses import dataclass

from .detection import METADATA_DUMP_FILE, detect_mcp, detect_thinking, dump_all_fields
from .gguf_fields import (
    FILE_TYPE_MAP,
    get_int_list,
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

# Pure SSM and hybrid SSM/attention architectures. Their KV cache only covers
# the attention layers, identified by zeros in a per-layer head_count_kv
# array or by attention.recurrent_layers/full_attention_interval (see
# _kv_heads_per_layer); where neither exists (pure SSMs), n_kv_heads is None
# and there is no KV cache at all.
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

# Built-in sliding-window layer pattern, (period, dense_first), for
# architectures whose GGUFs often carry attention.sliding_window but no
# sliding_window_pattern (Gemma3 and gpt-oss files never have one). Mirrors
# the load_swa_pattern/set_swa_pattern defaults in llama.cpp src/models/*.cpp;
# see gguf_fields.swa_pattern_from_period for the period semantics. Any other
# architecture without a pattern gets no SWA, as in llama.cpp (period 1 =
# explicitly no local layers, e.g. phi3, whose window llama.cpp ignores).
SWA_DEFAULT_PATTERN = {
    "gemma2": (2, False), "gemma3": (6, False), "gemma3n": (5, False),
    "gpt-oss": (2, False), "cohere2": (4, False), "cohere2moe": (4, True),
    "exaone4": (4, False), "exaone-moe": (4, False), "olmo2": (4, False),
    "plamo3": (8, False), "mellum": (4, False), "muse-glimmer": (4, False),
    "afmoe": (4, False), "smallthinker": (4, True), "laguna": (4, True),
    "phi3": (1, False),
}

# Hybrid architectures that mark their attention layers by an interval
# (every n-th layer is full attention, the rest recurrent) rather than by
# zeros in a per-layer head_count_kv array; the default applies when the GGUF
# carries neither attention.recurrent_layers nor full_attention_interval.
FULL_ATTENTION_INTERVAL_DEFAULT = {
    "qwen3next": 4, "qwen35": 4, "qwen35moe": 4, "qwen4exp": None, "minimax-01": 8,
}

# Architectures where only the first n layers keep a KV cache of their own
# and the rest reuse it, without an attention.shared_kv_layers key to say so
# (llama.cpp hard-codes n_layer_kv_from_start for them).
KV_LAYERS_FROM_START_DEFAULT = {"gemma3n": 20}

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
# Value tensor (head_dim_k and head_dim_v wide per KV head).
KV_BYTES_PER_ELEMENT = 2


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
    n_kv_heads: int | None  # None => pure SSM: no classic per-head KV cache
    mla_kv_dim: int | None = None
    # Sliding-window attention (Gemma2/3/4, Cohere2, gpt-oss, ...): the layers
    # flagged True in swa_pattern cache only swa_window tokens each,
    # regardless of context length, instead of scaling with the full context
    # like the rest. Ignored unless swa_window is set.
    swa_window: int | None = None
    swa_pattern: tuple[bool, ...] | None = None
    # One KV-head count per layer, overriding n_kv_heads where the model
    # varies it (Gemma 4: 8 on local, 2 on global layers). 0 marks a layer
    # without a KV cache of its own: the recurrent layers of a hybrid model,
    # or a layer reusing an earlier layer's cache (Gemma 3n/4 shared KV).
    kv_heads_per_layer: tuple[int, ...] | None = None
    # attention.key_length/value_length(_swa). llama.cpp falls back to
    # n_embd / n_heads, which is wrong for most current models (Qwen3,
    # Gemma 3/4, gpt-oss), and the SWA dims default to the full-layer ones.
    head_dim_k: int | None = None
    head_dim_v: int | None = None
    head_dim_k_swa: int | None = None
    head_dim_v_swa: int | None = None

    @property
    def head_dims(self):
        """Resolved (k, v, k_swa, v_swa) head dimensions."""
        fallback = self.n_embd // (self.n_heads or 1)
        k = self.head_dim_k or fallback
        v = self.head_dim_v or fallback
        return k, v, self.head_dim_k_swa or k, self.head_dim_v_swa or v

    def _is_swa(self, il):
        return bool(self.swa_window and self.swa_pattern and il < len(self.swa_pattern) and self.swa_pattern[il])

    def _kv_heads(self, il):
        if self.kv_heads_per_layer is not None:
            return self.kv_heads_per_layer[il] if il < len(self.kv_heads_per_layer) else 0
        return self.n_kv_heads or 0

    @property
    def swa_layers(self):
        return sum(1 for il in range(self.n_layers) if self._is_swa(il))

    @property
    def kv_layers(self):
        """Number of layers that keep a KV cache of their own."""
        return sum(1 for il in range(self.n_layers) if self._kv_heads(il))

    def _kv_bytes(self, swa):
        """Per-context-token KV bytes summed over the local (swa=True) or the
        full/global (swa=False) attention layers."""
        k, v, k_swa, v_swa = self.head_dims
        total = 0
        for il in range(self.n_layers):
            if self._is_swa(il) != swa:
                continue
            # K and V can differ in width (value_length != key_length), so
            # the two cached tensors are sized separately.
            dims = k_swa + v_swa if swa else k + v
            total += self._kv_heads(il) * dims * KV_BYTES_PER_ELEMENT
        return total

    def kv_bytes_per_ctx_token(self):
        """Bytes of KV-cache per context token contributed by the model's
        full/global-attention layers (all layers, unless swa_pattern carves
        some out into kv_bytes_per_ctx_token_swa instead)."""
        if self.mla_kv_dim:
            # MLA architectures (DeepSeek2, GLM-DSA, Mistral4, MiniCPM3) cache
            # a single shared compressed vector per token/layer instead of a
            # value per KV-head. They don't combine with sliding-window
            # attention in practice, so swa_pattern is ignored here.
            return self.n_layers * self.mla_kv_dim * KV_BYTES_PER_ELEMENT
        return self._kv_bytes(swa=False)

    def kv_bytes_per_ctx_token_swa(self):
        """Bytes of KV-cache per context token contributed by the model's
        local/sliding-window-attention layers, to be multiplied by
        min(ctx, swa_window) rather than the raw context length."""
        if self.mla_kv_dim or not self.swa_window:
            return 0
        return self._kv_bytes(swa=True)


def _recurrent_layers(reader, arch, n_layers):
    """One flag per layer, True for the recurrent (SSM/linear-attention)
    layers of a hybrid that marks them by attention.recurrent_layers or a
    full_attention_interval (Qwen3-Next/3.5, MiniMax-01); None otherwise."""
    flags = get_int_list(reader, f"{arch}.attention.recurrent_layers")
    if flags is not None:
        if len(flags) == 1:
            return [bool(flags[0])] * n_layers
        flags = [bool(v) for v in flags[:n_layers]]
        return flags + [False] * (n_layers - len(flags))
    arch_lower = arch.lower()
    if arch_lower not in FULL_ATTENTION_INTERVAL_DEFAULT:
        return None
    interval = (
        get_safe_int(reader, f"{arch}.full_attention_interval")
        or FULL_ATTENTION_INTERVAL_DEFAULT[arch_lower]
    )
    if not interval:
        return None
    return [(i + 1) % interval != 0 for i in range(n_layers)]


def _kv_heads_per_layer(reader, arch, n_layers, n_kv_heads, is_ssm):
    """Per-layer KV-head counts (see ModelShape.kv_heads_per_layer), with 0
    for every layer that keeps no KV cache of its own. None when all layers
    share n_kv_heads and cache their own KV (the plain case), and for SSM
    architectures whose attention layers can't be identified."""
    heads = get_int_list(reader, f"{arch}.attention.head_count_kv")
    varies = heads is not None and len(heads) > 1
    if heads is not None and varies:
        # Gemma 4 varies the count between local and global layers; hybrids
        # (Jamba, LFM2, Granite-Hybrid, Nemotron-H, ...) put 0 on their
        # recurrent layers.
        per_layer = (heads[:n_layers] + [0] * n_layers)[:n_layers]
    elif n_kv_heads:
        per_layer = [n_kv_heads] * n_layers
    else:
        return None

    recurrent = _recurrent_layers(reader, arch, n_layers) if is_ssm else None
    if recurrent is not None:
        per_layer = [0 if r else h for r, h in zip(recurrent, per_layer)]
        varies = True
    elif is_ssm and not varies:
        return None

    shared = get_safe_int(reader, f"{arch}.attention.shared_kv_layers")
    kv_from_start = n_layers - shared if shared else KV_LAYERS_FROM_START_DEFAULT.get(arch.lower())
    if kv_from_start is not None and 0 <= kv_from_start < n_layers:
        per_layer[kv_from_start:] = [0] * (n_layers - kv_from_start)
        varies = True

    return tuple(per_layer) if varies else None


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

    raw_kv = get_nonneg_int(reader,
        f"{arch}.attention.head_count_kv",
        f"{arch}.num_key_value_heads",
        f"{arch}.attention.kv_head_count",
    )
    is_ssm = arch_lower in SSM_ARCHS
    if not is_ssm and raw_kv == 0:
        # 0 means "same as n_heads" in llama.cpp convention
        raw_kv = n_heads
    kv_heads_per_layer = _kv_heads_per_layer(reader, arch, n_layers or 0, raw_kv, is_ssm)
    if kv_heads_per_layer is not None:
        # Representative value for display; the per-layer counts drive the
        # KV-cache size.
        n_kv_heads = max(kv_heads_per_layer, default=0) or None
    else:
        # A hybrid whose attention layers can't be told apart counts as pure
        # SSM (no KV cache) rather than as all-attention.
        n_kv_heads = None if is_ssm else raw_kv

    mla_kv_dim = None
    if arch_lower in MLA_ARCHS:
        kv_lora_rank = get_safe_int(reader, f"{arch}.attention.kv_lora_rank")
        rope_dim = get_safe_int(reader, f"{arch}.rope.dimension_count")
        if kv_lora_rank and rope_dim:
            mla_kv_dim = kv_lora_rank + rope_dim

    default_period, dense_first = SWA_DEFAULT_PATTERN.get(arch_lower, (None, False))
    swa_window, swa_pattern = get_sliding_window(reader, arch, n_layers or 0, default_period, dense_first)

    raw_name = clean_name(get_str(reader, "general.name"))
    name = resolve_name(raw_name, file_path)

    shape = ModelShape(
        n_layers=n_layers or 0,
        n_embd=n_embd or 0,
        n_heads=n_heads or 0,
        n_kv_heads=n_kv_heads,
        mla_kv_dim=mla_kv_dim,
        swa_window=swa_window,
        swa_pattern=swa_pattern,
        kv_heads_per_layer=kv_heads_per_layer,
        head_dim_k=get_safe_int(reader, f"{arch}.attention.key_length"),
        head_dim_v=get_safe_int(reader, f"{arch}.attention.value_length"),
        head_dim_k_swa=get_safe_int(reader, f"{arch}.attention.key_length_swa"),
        head_dim_v_swa=get_safe_int(reader, f"{arch}.attention.value_length_swa"),
    )
    head_dim_k, head_dim_v, head_dim_k_swa, head_dim_v_swa = shape.head_dims

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
        "head_dim_k": head_dim_k,
        "head_dim_v": head_dim_v,
        "head_dim_k_swa": head_dim_k_swa if swa_window else None,
        "head_dim_v_swa": head_dim_v_swa if swa_window else None,
        "kv_layers": shape.kv_layers,
        "kv_bytes_per_ctx_token": shape.kv_bytes_per_ctx_token(),
        "kv_bytes_per_ctx_token_swa": shape.kv_bytes_per_ctx_token_swa(),
        "swa_layers": shape.swa_layers,
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
