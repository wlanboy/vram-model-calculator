import json
import re

from vram_model_calculator import vram_calculator
from vram_model_calculator.vram_calculator import (
    calculate_vram_matrix,
    coerce_legacy_cache_int,
    get_color,
)

ANSI_RE = re.compile(r"\033\[[0-9;]*m")


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
