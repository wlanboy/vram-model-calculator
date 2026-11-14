import argparse
import json
import os

from ._model import MODEL_TYPE_LLM, is_pure_ssm

CACHE_FILE = "models_cache.json"

GPU_LIMITS = {
    "6GB (Entry)": 6.0,
    "12GB (Mid)": 12.0,
    "16GB (Pro)": 16.0,
    "24GB (Ultra)": 24.0
}

# A GPU is "tight" once usage crosses this fraction of its VRAM. Must match
# the constant in filter.js (checked by tests/test_vram_calculator.py).
TIGHT_FIT_RATIO = 0.85

# KV-Cache-Typen wie bei llama.cpp --cache-type-k/--cache-type-v. Der Cache
# speichert kv_bytes_per_ctx_token für f16 (2 Bytes/Element); die Faktoren
# rechnen das auf die Blockformate um: q8_0 = 34 Bytes pro 32 Elemente,
# q4_0 = 18 Bytes pro 32 Elemente, f16 = 64 Bytes pro 32 Elemente. Muss mit
# KV_CACHE_TYPES in filter.js übereinstimmen (geprüft von
# tests/test_vram_calculator.py).
KV_CACHE_TYPES = {
    "f16":  1.0,
    "q8_0": 34 / 64,
    "q4_0": 18 / 64,
}
KV_CACHE_TYPE_DEFAULT = "f16"

USECASES = {
    "Chat (8k)":    8000,
    "Code (32k)":   32000,
    "Doc (64k)":    64000,
    "Rev (128k)":   128000,
    "Res (256k)":   256000,
    "Agent (512k)": 512000,
    "Agent (1M)":   1000000,
}

def coerce_legacy_cache_int(val):
    """Reads an int field from models_cache.json, tolerating pre-typed
    entries: a single-character string is read as its ord() value (e.g. '('
    means 40), other strings are parsed as decimal, and anything else
    unparseable becomes 0."""
    if val is None:
        return 0
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        if len(val) == 1:
            return ord(val)
        try:
            return int(val)
        except ValueError:
            return 0
    return 0

def get_color(total, limit):
    if total <= limit * TIGHT_FIT_RATIO:
        return "\033[92m🟢\033[0m"  # Grün
    if total <= limit:
        return "\033[93m🟡\033[0m"  # Gelb
    return "\033[91m🔴\033[0m"      # Rot

def calculate_vram_matrix(cache_type=KV_CACHE_TYPE_DEFAULT):
    kv_factor = KV_CACHE_TYPES[cache_type]
    if not os.path.exists(CACHE_FILE):
        print("❌ Cache nicht gefunden.")
        return

    with open(CACHE_FILE, 'r') as f:
        loaded = json.load(f)

    # Versionseintrag und mmproj-Dateien überspringen
    models = {
        k: v for k, v in loaded.items()
        if isinstance(v, dict) and v.get("type") == MODEL_TYPE_LLM
    }

    print(f"KV-Cache-Typ: {cache_type}")

    for name, data in models.items():
        layers = coerce_legacy_cache_int(data.get("n_layers"))
        embd = coerce_legacy_cache_int(data.get("n_embd"))
        # Reine SSM-Modelle (Mamba, RWKV) haben n_kv_heads null oder 0 → kein KV-Cache
        is_ssm = is_pure_ssm(data.get("n_kv_heads"))
        # Recurrent-State von SSM- und Hybrid-Modellen: fest pro Session,
        # unabhängig vom Kontext und immer f32, also ohne kv_factor (siehe
        # RecurrentShape in _model.py).
        recurrent_state_gb = coerce_legacy_cache_int(data.get("recurrent_state_bytes")) / (1024**3)
        # Von _model.py vorberechnet (siehe ModelShape.kv_bytes_per_ctx_token/
        # _swa): einzige Quelle der KV-Cache-Formel. kv_bytes_per_ctx_token_swa
        # gilt nur bis swa_window Tokens (Sliding-Window-Attention-Layer wie
        # bei Gemma3/4, Cohere2, gpt-oss cachen nie mehr als das Fenster).
        kv_bytes_per_ctx_token = coerce_legacy_cache_int(data.get("kv_bytes_per_ctx_token"))
        kv_bytes_per_ctx_token_swa = coerce_legacy_cache_int(data.get("kv_bytes_per_ctx_token_swa"))
        swa_window = data.get("swa_window")
        base_size = data.get("file_size_gb", 0)

        if layers == 0 or embd == 0:
            continue

        arch = data.get("arch", "unknown")
        n_experts = data.get("n_experts")
        n_experts_used = data.get("n_experts_used")
        moe_tag = f" MoE {n_experts_used}/{n_experts}" if n_experts else ""

        print(f"\n\033[1m🤖 {name[:60]}\033[0m")
        state_tag = f" | Recurrent-State: {recurrent_state_gb * 1024:.1f} MB" if recurrent_state_gb else ""
        print(f"Arch: {arch}{moe_tag} | Size: {base_size:.2f} GB{state_tag}")

        header = f"{'Usecase':<12} | {'KV-Cache':<10} | " + " | ".join([f"{k:<10}" for k in GPU_LIMITS])
        print(header)
        print("-" * len(header))

        for uc_name, ctx in USECASES.items():
            kv_vram_bytes = kv_bytes_per_ctx_token * ctx
            if swa_window:
                kv_vram_bytes += kv_bytes_per_ctx_token_swa * min(ctx, swa_window)
            kv_vram = kv_vram_bytes * kv_factor / (1024**3) + recurrent_state_gb
            total = base_size + kv_vram

            status_row = []
            for limit in GPU_LIMITS.values():
                icon = get_color(total, limit)
                status_row.append(f"{icon} {total:>5.1f}G")

            kv_label = f"SSM {kv_vram:>4.2f}" if is_ssm else f"{kv_vram:>8.2f}"
            print(f"{uc_name:<12} | {kv_label}GB | " + " | ".join(status_row))


def add_cache_type_argument(parser):
    parser.add_argument(
        "--cache-type", choices=list(KV_CACHE_TYPES), default=KV_CACHE_TYPE_DEFAULT,
        help="KV-Cache-Typ wie bei llama.cpp --cache-type-k/-v (Default: %(default)s)",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="VRAM-Matrix aus models_cache.json berechnen und ausgeben.")
    add_cache_type_argument(parser)
    args = parser.parse_args(argv)
    calculate_vram_matrix(args.cache_type)


if __name__ == "__main__":
    main()
