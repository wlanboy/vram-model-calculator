import json
import re

from vram_model_calculator import vram_calculator
from vram_model_calculator.vram_calculator import (
    calculate_vram_matrix,
    get_color,
    to_int,
)

ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def strip_ansi(text):
    return ANSI_RE.sub("", text)


class TestToInt:
    def test_none_is_zero(self):
        assert to_int(None) == 0

    def test_int_passthrough(self):
        assert to_int(42) == 42

    def test_single_char_string_is_ord(self):
        assert to_int("(") == 40

    def test_numeric_string_is_parsed(self):
        assert to_int("128") == 128

    def test_non_numeric_multichar_string_is_zero(self):
        assert to_int("abc") == 0

    def test_other_types_are_zero(self):
        assert to_int(3.5) == 0
        assert to_int([1, 2]) == 0


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
                "n_heads": 32,
                "n_kv_heads": 8,
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
                "n_heads": 16,
                "n_kv_heads": 4,
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

    def test_mla_model_uses_combined_kv_dim_not_heads_formula(self, tmp_path, monkeypatch, capsys):
        # A DeepSeek2-style model where the naive n_kv_heads * head_dim formula
        # (32 * 128 = 4096) would hugely overstate the real MLA cache dim (576).
        cache = {
            "_version": 1,
            "MlaModel": {
                "type": "llm",
                "arch": "deepseek2",
                "n_layers": 32,
                "n_embd": 4096,
                "n_heads": 32,
                "n_kv_heads": 32,
                "mla_kv_dim": 576,  # kv_lora_rank(512) + rope.dimension_count(64)
                "file_size_gb": 1.0,
            },
        }
        cache_file = tmp_path / "cache.json"
        cache_file.write_text(json.dumps(cache))
        monkeypatch.setattr(vram_calculator, "CACHE_FILE", str(cache_file))

        calculate_vram_matrix()
        out = strip_ansi(capsys.readouterr().out)

        ctx = vram_calculator.USECASES["Chat (8k)"]
        expected_kv = (32 * 576 * ctx * vram_calculator.KV_BYTES_PER_ELEMENT) / (1024**3)
        naive_kv = (
            vram_calculator.KV_TENSORS_PER_LAYER * 32 * 32 * (4096 // 32) * ctx
            * vram_calculator.KV_BYTES_PER_ELEMENT
        ) / (1024**3)

        assert f"{expected_kv:>8.2f}" in out
        assert f"{naive_kv:>8.2f}" not in out

    def test_mla_arch_with_falsy_kv_dim_falls_back_to_heads_formula(self, tmp_path, monkeypatch, capsys):
        # If mla_kv_dim couldn't be determined (0/null), fall back to the
        # generic n_kv_heads formula instead of silently reporting 0 VRAM.
        cache = {
            "_version": 1,
            "MlaModelNoLoraRank": {
                "type": "llm",
                "arch": "deepseek2",
                "n_layers": 32,
                "n_embd": 4096,
                "n_heads": 32,
                "n_kv_heads": 32,
                "mla_kv_dim": None,
                "file_size_gb": 1.0,
            },
        }
        cache_file = tmp_path / "cache.json"
        cache_file.write_text(json.dumps(cache))
        monkeypatch.setattr(vram_calculator, "CACHE_FILE", str(cache_file))

        calculate_vram_matrix()
        out = strip_ansi(capsys.readouterr().out)

        ctx = vram_calculator.USECASES["Chat (8k)"]
        naive_kv = (
            vram_calculator.KV_TENSORS_PER_LAYER * 32 * 32 * (4096 // 32) * ctx
            * vram_calculator.KV_BYTES_PER_ELEMENT
        ) / (1024**3)
        assert f"{naive_kv:>8.2f}" in out

    def test_ssm_model_shows_ssm_label_and_no_kv_growth(self, tmp_path, monkeypatch, capsys):
        cache = {
            "_version": 1,
            "SsmModel": {
                "type": "llm",
                "arch": "mamba",
                "n_layers": 24,
                "n_embd": 2048,
                "n_heads": 16,
                "n_kv_heads": None,
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
