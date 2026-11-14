import json
import re
from pathlib import Path

from vram_model_calculator import vram_calculator
from vram_model_calculator.vram_calculator import (
    calculate_vram_matrix,
    coerce_legacy_cache_int,
    get_color,
)

ANSI_RE = re.compile(r"\033\[[0-9;]*m")
FILTER_JS = Path(__file__).resolve().parent.parent / "filter.js"


def strip_ansi(text):
    return ANSI_RE.sub("", text)


class TestCoerceLegacyCacheInt:
    def test_none_is_zero(self):
        assert coerce_legacy_cache_int(None) == 0

    def test_int_passthrough(self):
        assert coerce_legacy_cache_int(42) == 42

    def test_single_char_string_is_ord(self):
        assert coerce_legacy_cache_int("(") == 40

    def test_numeric_string_is_parsed(self):
        assert coerce_legacy_cache_int("128") == 128

    def test_non_numeric_multichar_string_is_zero(self):
        assert coerce_legacy_cache_int("abc") == 0

    def test_other_types_are_zero(self):
        assert coerce_legacy_cache_int(3.5) == 0
        assert coerce_legacy_cache_int([1, 2]) == 0


class TestGetColor:
    def test_green_when_well_under_limit(self):
        assert "92m" in get_color(total=5, limit=10)

    def test_yellow_when_tight(self):
        # TIGHT_FIT_RATIO is 0.85, so 9/10 = 0.9 is over the tight ratio but at/under limit
        assert "93m" in get_color(total=9, limit=10)

    def test_red_when_over_limit(self):
        assert "91m" in get_color(total=11, limit=10)

    def test_boundary_at_tight_ratio_is_green(self):
        assert "92m" in get_color(total=8.5, limit=10)

    def test_boundary_at_limit_is_yellow(self):
        assert "93m" in get_color(total=10, limit=10)


class TestCalculateVramMatrix:
    def test_missing_cache_file_prints_error(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(vram_calculator, "CACHE_FILE", str(tmp_path / "nope.json"))
        calculate_vram_matrix()
        assert "Cache nicht gefunden" in capsys.readouterr().out

    def test_skips_non_llm_entries_and_zero_layers(self, tmp_path, monkeypatch, capsys):
        cache = {
            "_version": 1,
            "an-adapter": {"type": "adapter", "name": "lora"},
            "zero-layers": {"type": "llm", "n_layers": 0, "n_embd": 4096},
        }
        cache_file = tmp_path / "cache.json"
        cache_file.write_text(json.dumps(cache))
        monkeypatch.setattr(vram_calculator, "CACHE_FILE", str(cache_file))

        calculate_vram_matrix()
        out = capsys.readouterr().out
        assert "an-adapter" not in out
        assert "zero-layers" not in out

    def test_renders_matrix_for_llm_entry(self, tmp_path, monkeypatch, capsys):
        cache = {
            "_version": 1,
            "MyModel": {
                "type": "llm",
                "arch": "llama",
                "n_layers": 32,
                "n_embd": 4096,
                "n_kv_heads": 8,
                "kv_bytes_per_ctx_token": 131072,
                "file_size_gb": 4.5,
            },
        }
        cache_file = tmp_path / "cache.json"
        cache_file.write_text(json.dumps(cache))
        monkeypatch.setattr(vram_calculator, "CACHE_FILE", str(cache_file))

        calculate_vram_matrix()
        out = strip_ansi(capsys.readouterr().out)
        assert "MyModel" in out
        assert "Arch: llama" in out
        assert "Chat (8k)" in out
        assert "Agent (1M)" in out

    def test_moe_tag_shown_when_experts_present(self, tmp_path, monkeypatch, capsys):
        cache = {
            "_version": 1,
            "MoeModel": {
                "type": "llm",
                "arch": "qwen3moe",
                "n_layers": 24,
                "n_embd": 2048,
                "n_kv_heads": 4,
                "kv_bytes_per_ctx_token": 49152,
                "file_size_gb": 8.0,
                "n_experts": 8,
                "n_experts_used": 2,
            },
        }
        cache_file = tmp_path / "cache.json"
        cache_file.write_text(json.dumps(cache))
        monkeypatch.setattr(vram_calculator, "CACHE_FILE", str(cache_file))

        calculate_vram_matrix()
        out = strip_ansi(capsys.readouterr().out)
        assert "MoE 2/8" in out

    def test_kv_vram_is_precomputed_bytes_times_ctx(self, tmp_path, monkeypatch, capsys):
        # The KV-cache formula itself (classic vs. MLA vs. SSM) lives entirely
        # in _model.py's ModelShape now; calculate_vram_matrix's only job is
        # to multiply the cache's precomputed kv_bytes_per_ctx_token by ctx.
        cache = {
            "_version": 1,
            "AnyModel": {
                "type": "llm",
                "arch": "deepseek2",
                "n_layers": 32,
                "n_embd": 4096,
                "n_kv_heads": 32,
                "kv_bytes_per_ctx_token": 36864,  # e.g. 32 layers * 576 mla_kv_dim * 2 bytes
                "file_size_gb": 1.0,
            },
        }
        cache_file = tmp_path / "cache.json"
        cache_file.write_text(json.dumps(cache))
        monkeypatch.setattr(vram_calculator, "CACHE_FILE", str(cache_file))

        calculate_vram_matrix()
        out = strip_ansi(capsys.readouterr().out)

        ctx = vram_calculator.USECASES["Chat (8k)"]
        expected_kv = (36864 * ctx) / (1024**3)
        assert f"{expected_kv:>8.2f}" in out

    def test_swa_kv_vram_caps_local_layers_at_window(self, tmp_path, monkeypatch, capsys):
        # 1 global layer (scales with ctx) + 5 local/SWA layers (capped at
        # swa_window=1024 tokens regardless of the usecase's context length).
        cache = {
            "_version": 1,
            "Gemma3Like": {
                "type": "llm",
                "arch": "gemma3",
                "n_layers": 6,
                "n_embd": 4096,
                "n_kv_heads": 8,
                "kv_bytes_per_ctx_token": 2048,       # 1 global layer's per-token bytes
                "kv_bytes_per_ctx_token_swa": 10240,  # 5 local layers' per-token bytes
                "swa_window": 1024,
                "file_size_gb": 1.0,
            },
        }
        cache_file = tmp_path / "cache.json"
        cache_file.write_text(json.dumps(cache))
        monkeypatch.setattr(vram_calculator, "CACHE_FILE", str(cache_file))

        calculate_vram_matrix()
        out = strip_ansi(capsys.readouterr().out)

        ctx = vram_calculator.USECASES["Doc (64k)"]  # far beyond the 1024-token window
        expected_kv = (2048 * ctx + 10240 * 1024) / (1024**3)
        assert f"{expected_kv:>8.2f}" in out

    def test_quantized_kv_cache_scales_precomputed_f16_bytes(self, tmp_path, monkeypatch, capsys):
        cache = {
            "_version": 1,
            "AnyModel": {
                "type": "llm",
                "arch": "llama",
                "n_layers": 32,
                "n_embd": 4096,
                "n_kv_heads": 8,
                "kv_bytes_per_ctx_token": 131072,
                "kv_bytes_per_ctx_token_swa": 65536,
                "swa_window": 4096,
                "file_size_gb": 1.0,
            },
        }
        cache_file = tmp_path / "cache.json"
        cache_file.write_text(json.dumps(cache))
        monkeypatch.setattr(vram_calculator, "CACHE_FILE", str(cache_file))

        calculate_vram_matrix("q8_0")
        out = strip_ansi(capsys.readouterr().out)

        assert "KV-Cache-Typ: q8_0" in out
        ctx = vram_calculator.USECASES["Doc (64k)"]
        f16_kv = (131072 * ctx + 65536 * 4096) / (1024**3)
        assert f"{f16_kv * 34 / 64:>8.2f}" in out
        assert f"{f16_kv:>8.2f}" not in out

    def test_main_passes_cache_type_option(self, monkeypatch):
        calls = []
        monkeypatch.setattr(vram_calculator, "calculate_vram_matrix", calls.append)
        vram_calculator.main(["--cache-type", "q4_0"])
        vram_calculator.main([])
        assert calls == ["q4_0", "f16"]

    def test_ssm_model_shows_ssm_label_and_no_kv_growth(self, tmp_path, monkeypatch, capsys):
        cache = {
            "_version": 1,
            "SsmModel": {
                "type": "llm",
                "arch": "mamba",
                "n_layers": 24,
                "n_embd": 2048,
                "n_kv_heads": None,
                "kv_bytes_per_ctx_token": 0,
                "file_size_gb": 2.0,
            },
        }
        cache_file = tmp_path / "cache.json"
        cache_file.write_text(json.dumps(cache))
        monkeypatch.setattr(vram_calculator, "CACHE_FILE", str(cache_file))

        calculate_vram_matrix()
        out = strip_ansi(capsys.readouterr().out)
        assert "SSM" in out
        # SSM models have no KV growth, so every usecase row should report the base size.
        assert out.count("2.0G") >= len(vram_calculator.USECASES)

    def test_zero_kv_heads_counts_as_ssm_like_filter_js(self, tmp_path, monkeypatch, capsys):
        cache = {
            "_version": 1,
            "ZeroKv": {
                "type": "llm",
                "arch": "mamba",
                "n_layers": 24,
                "n_embd": 2048,
                "n_kv_heads": 0,
                "kv_bytes_per_ctx_token": 0,
                "file_size_gb": 2.0,
            },
        }
        cache_file = tmp_path / "cache.json"
        cache_file.write_text(json.dumps(cache))
        monkeypatch.setattr(vram_calculator, "CACHE_FILE", str(cache_file))

        calculate_vram_matrix()
        out = strip_ansi(capsys.readouterr().out)
        assert out.count("SSM 0.00GB") == len(vram_calculator.USECASES)

    def test_recurrent_state_adds_to_total_unscaled_by_cache_type(self, tmp_path, monkeypatch, capsys):
        # 256 MiB recurrent state (always f32 in llama.cpp) plus a hybrid's
        # attention-layer KV cache, which q4_0 does shrink.
        state_bytes = 256 * 1024**2
        cache = {
            "_version": 1,
            "Hybrid": {
                "type": "llm",
                "arch": "qwen35",
                "n_layers": 24,
                "n_embd": 1024,
                "n_kv_heads": 2,
                "kv_bytes_per_ctx_token": 12288,
                "recurrent_state_bytes": state_bytes,
                "file_size_gb": 1.0,
            },
        }
        cache_file = tmp_path / "cache.json"
        cache_file.write_text(json.dumps(cache))
        monkeypatch.setattr(vram_calculator, "CACHE_FILE", str(cache_file))

        calculate_vram_matrix("q4_0")
        out = strip_ansi(capsys.readouterr().out)

        assert "Recurrent-State: 256.0 MB" in out
        ctx = vram_calculator.USECASES["Code (32k)"]
        per_session = 12288 * ctx * 18 / 64 / (1024**3) + 0.25
        assert f"{per_session:>8.2f}GB" in out
        assert f"{1.0 + per_session:>5.1f}G" in out

    def test_pure_ssm_shows_recurrent_state(self, tmp_path, monkeypatch, capsys):
        cache = {
            "_version": 1,
            "Mamba": {
                "type": "llm",
                "arch": "mamba",
                "n_layers": 64,
                "n_embd": 2560,
                "n_kv_heads": None,
                "kv_bytes_per_ctx_token": 0,
                "recurrent_state_bytes": 512 * 1024**2,
                "file_size_gb": 2.0,
            },
        }
        cache_file = tmp_path / "cache.json"
        cache_file.write_text(json.dumps(cache))
        monkeypatch.setattr(vram_calculator, "CACHE_FILE", str(cache_file))

        calculate_vram_matrix()
        out = strip_ansi(capsys.readouterr().out)
        assert out.count("SSM 0.50GB") == len(vram_calculator.USECASES)
        assert out.count("2.5G") >= len(vram_calculator.USECASES)


class TestConstantsInSyncWithFilterJs:
    def test_tight_fit_ratio_matches(self):
        match = re.search(r"const\s+TIGHT_FIT_RATIO\s*=\s*([0-9.]+)\s*;", FILTER_JS.read_text(encoding="utf-8"))
        assert match, "TIGHT_FIT_RATIO not found in filter.js"
        assert float(match.group(1)) == vram_calculator.TIGHT_FIT_RATIO

    def test_kv_cache_types_match(self):
        source = FILTER_JS.read_text(encoding="utf-8")
        block = re.search(r"const\s+KV_CACHE_TYPES\s*=\s*\{(.*?)\};", source, re.DOTALL)
        assert block, "KV_CACHE_TYPES not found in filter.js"
        js_types = {
            name: float(num) / float(den or 1)
            for name, num, den in re.findall(r"(\w+)\s*:\s*([0-9.]+)(?:\s*/\s*([0-9.]+))?", block.group(1))
        }
        assert js_types == vram_calculator.KV_CACHE_TYPES

    def test_kv_cache_type_default_matches(self):
        match = re.search(r'const\s+KV_CACHE_TYPE_DEFAULT\s*=\s*"(\w+)"\s*;', FILTER_JS.read_text(encoding="utf-8"))
        assert match, "KV_CACHE_TYPE_DEFAULT not found in filter.js"
        assert match.group(1) == vram_calculator.KV_CACHE_TYPE_DEFAULT
