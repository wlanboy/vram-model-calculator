import json
import os

CACHE_FILE = "models_cache.json"

GPU_LIMITS = {
    "6GB (Entry)": 6.0,
    "12GB (Mid)": 12.0,
    "16GB (Pro)": 16.0,
    "24GB (Ultra)": 24.0
}

USECASES = {
    "Chat (8k)":    8000,
    "Code (32k)":   32000,
    "Doc (64k)":    64000,
    "Rev (128k)":   128000,
    "Res (256k)":   256000,
    "Agent (512k)": 512000,
    "Agent (1M)":   1000000,
}

def to_int(val):
    """Konvertiert ASCII-Steuerzeichen oder Strings sicher in Integer."""
    if val is None: return 0
    if isinstance(val, int): return val
    if isinstance(val, str):
        if len(val) == 1: return ord(val) # Wandelt z.B. '(' in 40 um
        try: return int(val)
        except: return 0
    return 0

def get_color(total, limit):
    if total <= limit * 0.85: return "\033[92m🟢\033[0m" # Grün
    if total <= limit: return "\033[93m🟡\033[0m"        # Gelb
    return "\033[91m🔴\033[0m"                           # Rot

def calculate_vram_matrix():
    if not os.path.exists(CACHE_FILE):
        print("❌ Cache nicht gefunden.")
        return

    with open(CACHE_FILE, 'r') as f:
        loaded = json.load(f)

    # Versionseintrag und mmproj-Dateien überspringen
    models = {
        k: v for k, v in loaded.items()
        if isinstance(v, dict) and v.get("type") == "llm"
    }

    for name, data in models.items():
        layers = to_int(data.get("n_layers"))
        embd = to_int(data.get("n_embd"))
        heads = to_int(data.get("n_heads"))
        # SSM-Modelle (LFM2, Nemotron-H) haben n_kv_heads=0 → kein KV-Cache
        kv_heads = to_int(data.get("n_kv_heads"))
        base_size = data.get("file_size_gb", 0)

        if layers == 0 or embd == 0: continue

        arch = data.get("arch", "unknown")
        n_experts = data.get("n_experts")
        n_experts_used = data.get("n_experts_used")
        moe_tag = f" MoE {n_experts_used}/{n_experts}" if n_experts else ""

        print(f"\n\033[1m🤖 {name[:60]}\033[0m")
        print(f"Arch: {arch}{moe_tag} | Size: {base_size:.2f} GB")
        
        header = f"{'Usecase':<12} | {'KV-Cache':<10} | " + " | ".join([f"{k:<10}" for k in GPU_LIMITS.keys()])
        print(header)
        print("-" * len(header))

        for uc_name, ctx in USECASES.items():
            # KV-Cache: bei SSM-Modellen (kv_heads=0) entfällt der KV-Cache
            head_dim = embd // (heads if heads > 0 else 1)
            kv_dim = kv_heads * head_dim
            kv_vram = (2 * layers * kv_dim * ctx * 2) / (1024**3) if kv_heads > 0 else 0.0
            
            total = base_size + kv_vram
            
            status_row = []
            for _, limit in GPU_LIMITS.items():
                icon = get_color(total, limit)
                status_row.append(f"{icon} {total:>5.1f}G")
            
            print(f"{uc_name:<12} | {kv_vram:>8.2f}GB | " + " | ".join(status_row))

if __name__ == "__main__":
    calculate_vram_matrix()