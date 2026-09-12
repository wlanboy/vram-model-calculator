import json
import os

from ._model import MODEL_TYPE_LLM

CACHE_FILE = "models_cache.json"

GPU_LIMITS = {
    "6GB (Entry)": 6.0,
    "12GB (Mid)": 12.0,
    "16GB (Pro)": 16.0,
    "24GB (Ultra)": 24.0
}

# A GPU is "tight" once usage crosses this fraction of its VRAM. Keep in sync
# with the identical constant in filter.js.
TIGHT_FIT_RATIO = 0.85

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

def calculate_vram_matrix():
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

    for name, data in models.items():
        layers = coerce_legacy_cache_int(data.get("n_layers"))
        embd = coerce_legacy_cache_int(data.get("n_embd"))
        # SSM-Modelle (LFM2, Nemotron-H) haben n_kv_heads=null → kein klassischer KV-Cache
        is_ssm = data.get("n_kv_heads") is None
        # Von _model.py vorberechnet (siehe ModelShape.kv_bytes_per_ctx_token):
        # einzige Quelle der KV-Cache-Formel, hier nur noch mit ctx multipliziert.
        kv_bytes_per_ctx_token = coerce_legacy_cache_int(data.get("kv_bytes_per_ctx_token"))
        base_size = data.get("file_size_gb", 0)

        if layers == 0 or embd == 0:
            continue

        arch = data.get("arch", "unknown")
        n_experts = data.get("n_experts")
        n_experts_used = data.get("n_experts_used")
        moe_tag = f" MoE {n_experts_used}/{n_experts}" if n_experts else ""

        print(f"\n\033[1m🤖 {name[:60]}\033[0m")
        print(f"Arch: {arch}{moe_tag} | Size: {base_size:.2f} GB")

        header = f"{'Usecase':<12} | {'KV-Cache':<10} | " + " | ".join([f"{k:<10}" for k in GPU_LIMITS])
        print(header)
        print("-" * len(header))

        for uc_name, ctx in USECASES.items():
            kv_vram = (kv_bytes_per_ctx_token * ctx) / (1024**3)
            total = base_size + kv_vram

            status_row = []
            for limit in GPU_LIMITS.values():
                icon = get_color(total, limit)
                status_row.append(f"{icon} {total:>5.1f}G")

            kv_label = "     SSM" if is_ssm else f"{kv_vram:>8.2f}"
            print(f"{uc_name:<12} | {kv_label}GB | " + " | ".join(status_row))


def main():
    calculate_vram_matrix()


if __name__ == "__main__":
    main()
