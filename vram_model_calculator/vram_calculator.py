"""VRAM requirement per model from models_cache.json: weights, KV cache,
recurrent state, compute buffer and runtime overhead. filter.js computes the
same in the browser; the shared constants are checked by
tests/test_vram_calculator.py."""
import argparse
import json
import os
from dataclasses import asdict, dataclass

from ._model import COMPUTE_MASK_BYTES_PER_CTX_TOKEN, MODEL_TYPE_LLM, is_pure_ssm
from .config import add_cache_argument, load_config, resolve_cache_file

GIB = 1024**3

DEFAULT_GPUS = [6.0, 12.0, 16.0, 24.0]
DEFAULT_CONTEXTS = [8192, 32768, 65536, 131072, 262144, 524288, 1048576]

# Driver/runtime context (CUDA, SYCL, Vulkan) plus small llama.cpp buffers
# (output buffer, graph metadata), per GPU. Must match filter.js.
DEFAULT_OVERHEAD_GB = 0.5

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

STATUS_GOOD = "good"
STATUS_TIGHT = "tight"
STATUS_NONE = "none"
STATUS_ICONS = {
    STATUS_GOOD:  "\033[92m🟢\033[0m",
    STATUS_TIGHT: "\033[93m🟡\033[0m",
    STATUS_NONE:  "\033[91m🔴\033[0m",
}


@dataclass(frozen=True)
class VramEstimate:
    """VRAM parts in GB. kv_gb and state_gb are per session."""

    weights_gb: float
    kv_gb: float
    state_gb: float
    compute_gb: float
    overhead_gb: float
    sessions: int

    @property
    def per_session_gb(self):
        return self.kv_gb + self.state_gb

    @property
    def total_gb(self):
        return self.weights_gb + self.per_session_gb * self.sessions + self.compute_gb + self.overhead_gb


def estimate(entry, ctx, sessions=1, cache_type=KV_CACHE_TYPE_DEFAULT, overhead_gb=DEFAULT_OVERHEAD_GB):
    """VRAM estimate for one cache entry at ctx tokens per session.

    In llama.cpp terms that is `-c <ctx * sessions> -np <sessions>`: every
    slot gets ctx tokens, the KV mask in the compute buffer covers all of them.
    """
    kv_bytes = (entry.get("kv_bytes_per_ctx_token") or 0) * ctx
    swa_window = entry.get("swa_window")
    if swa_window:
        # Sliding-window layers (Gemma 2/3/4, Cohere2, gpt-oss) never cache
        # more than the window.
        kv_bytes += (entry.get("kv_bytes_per_ctx_token_swa") or 0) * min(ctx, swa_window)
    compute_bytes = (entry.get("compute_buffer_bytes") or 0) + ctx * sessions * COMPUTE_MASK_BYTES_PER_CTX_TOKEN
    return VramEstimate(
        weights_gb=entry.get("file_size_gb") or 0,
        kv_gb=kv_bytes * KV_CACHE_TYPES[cache_type] / GIB,
        # Recurrent state of SSM and hybrid models: fixed per session and
        # always f32 in llama.cpp, so the KV cache type doesn't apply.
        state_gb=(entry.get("recurrent_state_bytes") or 0) / GIB,
        compute_gb=compute_bytes / GIB,
        overhead_gb=overhead_gb,
        sessions=sessions,
    )


def fit_status(total, limit):
    if total <= limit * TIGHT_FIT_RATIO:
        return STATUS_GOOD
    if total <= limit:
        return STATUS_TIGHT
    return STATUS_NONE


def get_color(total, limit):
    return STATUS_ICONS[fit_status(total, limit)]


_CTX_SUFFIXES = {"k": 1024, "m": 1024**2}


def parse_ctx(text):
    """'8k' -> 8192, '1M' -> 1048576, '32768' -> 32768."""
    t = text.strip().lower()
    factor = _CTX_SUFFIXES.get(t[-1:], 1)
    number = t[:-1] if factor != 1 else t
    try:
        value = int(float(number) * factor)
    except ValueError:
        raise argparse.ArgumentTypeError(f"ungültige Kontextlänge: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"Kontextlänge muss positiv sein: {text!r}")
    return value


def parse_ctx_list(text):
    return [parse_ctx(part) for part in text.split(",") if part.strip()]


def fmt_ctx(tokens):
    if tokens % 1024**2 == 0:
        return f"{tokens // 1024**2}M"
    if tokens % 1024 == 0:
        return f"{tokens // 1024}k"
    return str(tokens)


def fmt_gpu(gb):
    return f"{gb:g} GB"


def load_models(cache_file):
    """LLM entries with usable dimensions, as {key: entry}; None if the
    cache file doesn't exist."""
    if not os.path.exists(cache_file):
        return None
    with open(cache_file, 'r') as f:
        loaded = json.load(f)
    return {
        k: v for k, v in loaded.items()
        if isinstance(v, dict) and v.get("type") == MODEL_TYPE_LLM and v.get("n_layers") and v.get("n_embd")
    }


def _matches(key, entry, text):
    t = text.lower()
    return any(t in str(v).lower() for v in (key, entry.get("name", ""), entry.get("arch", "")))


def select_models(models, name_filter=None, fits=None, ctx=None, **est_kwargs):
    """Applies --filter (name/arch/key substring) and --fits (total at ctx
    within that many GB, tight fits included)."""
    selected = {}
    for key, entry in models.items():
        if name_filter and not _matches(key, entry, name_filter):
            continue
        if fits is not None and estimate(entry, ctx, **est_kwargs).total_gb > fits:
            continue
        selected[key] = entry
    return selected


def sessions_hint(ctx, sessions):
    if sessions == 1:
        return f"llama.cpp: -c {ctx}"
    return f"entspricht llama.cpp -c {ctx * sessions} -np {sessions}"


def print_matrix(models, contexts, gpus, est_kwargs):
    for key, entry in models.items():
        name = entry.get("name") or key
        arch = entry.get("arch", "unknown")
        n_experts = entry.get("n_experts")
        moe_tag = f" MoE {entry.get('n_experts_used')}/{n_experts}" if n_experts else ""
        base = estimate(entry, contexts[0], **est_kwargs)
        state_tag = f" | Recurrent-State: {base.state_gb * 1024:.1f} MB" if base.state_gb else ""
        quant = entry.get("quant") or "?"

        print(f"\n\033[1m🤖 {name[:60]}\033[0m")
        print(
            f"Arch: {arch}{moe_tag} | Quant: {quant} | Size: {base.weights_gb:.2f} GB"
            f" | Compute: {base.compute_gb:.2f} GB | Overhead: {base.overhead_gb:.2f} GB{state_tag}"
        )

        header = f"{'Kontext':<8} | {'KV-Cache':<10} | " + " | ".join(f"{fmt_gpu(g):<10}" for g in gpus)
        print(header)
        print("-" * len(header))

        ssm = is_pure_ssm(entry.get("n_kv_heads"))
        for ctx in contexts:
            est = estimate(entry, ctx, **est_kwargs)
            kv = est.per_session_gb * est.sessions
            cells = [f"{get_color(est.total_gb, g)} {est.total_gb:>5.1f}G" for g in gpus]
            kv_label = f"SSM {kv:>4.2f}" if ssm else f"{kv:>8.2f}"
            print(f"{fmt_ctx(ctx):<8} | {kv_label}GB | " + " | ".join(cells))


def print_compact(models, contexts, gpus, est_kwargs):
    for ctx in contexts:
        print(f"\n\033[1mKontext {fmt_ctx(ctx)}\033[0m ({sessions_hint(ctx, est_kwargs['sessions'])})")
        header = (
            f"{'Modell':<44} | {'Quant':<8} | {'Größe':>7} | {'KV':>7} | {'Gesamt':>7} | "
            + " | ".join(f"{fmt_gpu(g):^6}" for g in gpus)
        )
        print(header)
        print("-" * len(header))
        for key, entry in models.items():
            est = estimate(entry, ctx, **est_kwargs)
            name = (entry.get("name") or key)[:44]
            icons = " | ".join(f"{get_color(est.total_gb, g):^6}" for g in gpus)
            print(
                f"{name:<44} | {(entry.get('quant') or '?')[:8]:<8} | {est.weights_gb:>6.2f}G | "
                f"{est.per_session_gb * est.sessions:>6.2f}G | {est.total_gb:>6.2f}G | {icons}"
            )


def build_json(models, contexts, gpus, est_kwargs):
    result = []
    for key, entry in models.items():
        rows = []
        for ctx in contexts:
            est = estimate(entry, ctx, **est_kwargs)
            rows.append({
                "ctx": ctx,
                "llama_cpp": {"c": ctx * est.sessions, "np": est.sessions},
                **{k: round(v, 4) if isinstance(v, float) else v for k, v in asdict(est).items()},
                "total_gb": round(est.total_gb, 4),
                "fits": {fmt_gpu(g): fit_status(est.total_gb, g) for g in gpus},
            })
        result.append({
            "key": key,
            "name": entry.get("name") or key,
            "arch": entry.get("arch"),
            "quant": entry.get("quant"),
            "n_ctx_orig": entry.get("n_ctx_orig"),
            "contexts": rows,
        })
    return result


def calculate_vram_matrix(cache_type=KV_CACHE_TYPE_DEFAULT, cache_file=None, gpus=None, contexts=None,
                          sessions=1, overhead_gb=DEFAULT_OVERHEAD_GB, name_filter=None, fits=None,
                          output="matrix"):
    """Prints the VRAM table for all (or the filtered) models. output is
    'matrix' (one table per model), 'compact' (one row per model) or 'json'."""
    cache_file = cache_file or resolve_cache_file()
    gpus = gpus or DEFAULT_GPUS
    contexts = sorted(contexts or DEFAULT_CONTEXTS)
    est_kwargs = {"sessions": sessions, "cache_type": cache_type, "overhead_gb": overhead_gb}

    models = load_models(cache_file)
    if models is None:
        print(f"❌ Cache nicht gefunden: {cache_file}")
        return 1
    models = select_models(models, name_filter, fits, contexts[0], **est_kwargs)

    if output == "json":
        print(json.dumps(build_json(models, contexts, gpus, est_kwargs), indent=2, ensure_ascii=False))
        return 0

    print(f"KV-Cache-Typ: {cache_type} | Sessions: {sessions} | Overhead: {overhead_gb:g} GB | {len(models)} Modelle")
    if output == "compact":
        print_compact(models, contexts, gpus, est_kwargs)
    else:
        if sessions > 1:
            print(f"Kontext gilt pro Session ({sessions_hint(contexts[0], sessions)} usw.)")
        print_matrix(models, contexts, gpus, est_kwargs)
    return 0


def add_cache_type_argument(parser):
    parser.add_argument(
        "--cache-type", choices=list(KV_CACHE_TYPES), default=KV_CACHE_TYPE_DEFAULT,
        help="KV-Cache-Typ wie bei llama.cpp --cache-type-k/-v (Default: %(default)s)",
    )


def _positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("muss mindestens 1 sein")
    return value


def add_arguments(parser):
    add_cache_type_argument(parser)
    parser.add_argument(
        "--gpu", type=float, action="append", metavar="GB",
        help="GPU-Größe in GB, mehrfach möglich (Default: 6, 12, 16, 24)",
    )
    parser.add_argument(
        "--ctx", type=parse_ctx_list, action="append", metavar="LISTE",
        help="Kontextlängen pro Session, z. B. 8k,32k,128k (Default: 8k bis 1M in Zweierpotenzen)",
    )
    parser.add_argument("--sessions", type=_positive_int, default=1, help="Parallele Sessions (llama.cpp -np)")
    parser.add_argument(
        "--overhead", type=float, default=DEFAULT_OVERHEAD_GB, metavar="GB",
        help="Fester Overhead für Treiber und Laufzeit in GB (Default: %(default)s)",
    )
    parser.add_argument("--filter", dest="name_filter", metavar="TEXT", help="Nur Modelle, deren Name/Arch/Key TEXT enthält")
    parser.add_argument(
        "--fits", type=float, metavar="GB",
        help="Nur Modelle, die beim kleinsten Kontext in GB passen (knapp zählt mit)",
    )
    fmt = parser.add_mutually_exclusive_group()
    fmt.add_argument("--json", dest="output", action="store_const", const="json", help="Ausgabe als JSON")
    fmt.add_argument(
        "--compact", dest="output", action="store_const", const="compact",
        help="Eine Zeile pro Modell statt einer Matrix je Modell",
    )
    parser.set_defaults(output="matrix")
    add_cache_argument(parser)


def run(args):
    contexts = [c for group in args.ctx for c in group] if args.ctx else None
    return calculate_vram_matrix(
        cache_type=args.cache_type,
        cache_file=resolve_cache_file(args.cache, load_config()),
        gpus=args.gpu,
        contexts=contexts,
        sessions=args.sessions,
        overhead_gb=args.overhead,
        name_filter=args.name_filter,
        fits=args.fits,
        output=args.output,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="VRAM-Bedarf aus models_cache.json berechnen und ausgeben.")
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
