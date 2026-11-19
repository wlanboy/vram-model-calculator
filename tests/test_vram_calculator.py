import argparse
import json
import re
from pathlib import Path

import pytest

from vram_model_calculator import _model, vram_calculator
from vram_model_calculator.vram_calculator import (
    calculate_vram_matrix,
    estimate,
    fit_status,
    fmt_ctx,
    get_color,
    parse_ctx,
    parse_ctx_list,
)

ANSI_RE = re.compile(r"\033\[[0-9;]*m")
FILTER_JS = Path(__file__).resolve().parent.parent / "filter.js"
GIB = 1024**3


def strip_ansi(text):
    return ANSI_RE.sub("", text)


def write_cache(tmp_path, **entries):
    cache_file = tmp_path / "cache.json"
    cache_file.write_text(json.dumps({"_version": 1, **entries}))
    return str(cache_file)


def llm(**fields):
    return {"type": "llm", "arch": "llama", "n_layers": 32, "n_embd": 4096, "n_kv_heads": 8,
            "kv_bytes_per_ctx_token": 131072, "file_size_gb": 4.5, **fields}


class TestGetColor:
    def test_green_when_well_under_limit(self):
        assert "92m" in get_color(total=5, limit=10)

    def test_yellow_when_tight(self):
        # TIGHT_FIT_RATIO is 0.85, so 9/10 = 0.9 is over the tight ratio but at/under limit
        assert "93m" in get_color(total=9, limit=10)

    def test_red_when_over_limit(self):
        assert "91m" in get_color(total=11, limit=10)

    def test_boundary_at_tight_ratio_is_green(self):
        assert fit_status(8.5, 10) == "good"

    def test_boundary_at_limit_is_yellow(self):
        assert fit_status(10, 10) == "tight"


class TestParseCtx:
    @pytest.mark.parametrize("text,expected", [
        ("8k", 8192), ("32K", 32768), ("1M", 1048576), ("1.5k", 1536), ("4096", 4096),
    ])
    def test_values(self, text, expected):
        assert parse_ctx(text) == expected

    @pytest.mark.parametrize("text", ["", "abc", "0", "-8k"])
    def test_invalid(self, text):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_ctx(text)

    def test_list(self):
        assert parse_ctx_list("8k, 32k,128k") == [8192, 32768, 131072]

    def test_fmt_ctx_round_trip(self):
        assert [fmt_ctx(c) for c in (8192, 1048576, 1000)] == ["8k", "1M", "1000"]


class TestEstimate:
    def test_all_parts(self):
        entry = llm(kv_bytes_per_ctx_token=1000, recurrent_state_bytes=GIB // 4, compute_buffer_bytes=GIB // 2)
        est = estimate(entry, 8192, sessions=2, cache_type="q4_0", overhead_gb=0.5)
        assert est.weights_gb == 4.5
        assert est.kv_gb == pytest.approx(1000 * 8192 * 18 / 64 / GIB)
        assert est.state_gb == 0.25
        mask = 8192 * 2 * _model.COMPUTE_MASK_BYTES_PER_CTX_TOKEN
        assert est.compute_gb == pytest.approx((GIB // 2 + mask) / GIB)
        assert est.total_gb == pytest.approx(4.5 + (est.kv_gb + 0.25) * 2 + est.compute_gb + 0.5)

    def test_swa_layers_capped_at_window(self):
        entry = llm(kv_bytes_per_ctx_token=2048, kv_bytes_per_ctx_token_swa=10240, swa_window=1024)
        est = estimate(entry, 65536)
        assert est.kv_gb == pytest.approx((2048 * 65536 + 10240 * 1024) / GIB)

    def test_missing_fields_count_as_zero(self):
        est = estimate({"type": "llm"}, 8192, overhead_gb=0)
        assert est.total_gb == pytest.approx(8192 * _model.COMPUTE_MASK_BYTES_PER_CTX_TOKEN / GIB)


class TestCalculateVramMatrix:
    def test_missing_cache_file_prints_error(self, tmp_path, capsys):
        assert calculate_vram_matrix(cache_file=str(tmp_path / "nope.json")) == 1
        assert "Cache nicht gefunden" in capsys.readouterr().out

    def test_skips_non_llm_entries_and_zero_layers(self, tmp_path, capsys):
        cache = write_cache(tmp_path, **{
            "an-adapter": {"type": "adapter", "name": "lora"},
            "zero-layers": {"type": "llm", "n_layers": 0, "n_embd": 4096},
        })
        calculate_vram_matrix(cache_file=cache)
        out = capsys.readouterr().out
        assert "an-adapter" not in out
        assert "zero-layers" not in out

    def test_renders_matrix_with_default_contexts_and_gpus(self, tmp_path, capsys):
        calculate_vram_matrix(cache_file=write_cache(tmp_path, MyModel=llm(quant="Q4_K_M")))
        out = strip_ansi(capsys.readouterr().out)
        assert "MyModel" in out
        assert "Arch: llama | Quant: Q4_K_M" in out
        assert "Overhead: 0.50 GB" in out
        for label in ("8k ", "32k", "1M "):
            assert label in out
        for gpu in ("6 GB", "12 GB", "16 GB", "24 GB"):
            assert gpu in out

    def test_moe_tag_shown_when_experts_present(self, tmp_path, capsys):
        calculate_vram_matrix(cache_file=write_cache(tmp_path, M=llm(n_experts=8, n_experts_used=2)))
        assert "MoE 2/8" in strip_ansi(capsys.readouterr().out)

    def test_kv_column_and_total(self, tmp_path, capsys):
        entry = llm(kv_bytes_per_ctx_token=36864, file_size_gb=1.0, compute_buffer_bytes=GIB // 4)
        calculate_vram_matrix(cache_file=write_cache(tmp_path, M=entry), contexts=[8192], gpus=[24])
        out = strip_ansi(capsys.readouterr().out)
        est = estimate(entry, 8192)
        assert f"{est.kv_gb:>8.2f}GB" in out
        assert f"{est.total_gb:>5.1f}G" in out

    def test_quantized_kv_cache_scales_f16_bytes(self, tmp_path, capsys):
        calculate_vram_matrix("q8_0", cache_file=write_cache(tmp_path, M=llm()), contexts=[65536])
        out = strip_ansi(capsys.readouterr().out)
        assert "KV-Cache-Typ: q8_0" in out
        f16_kv = 131072 * 65536 / GIB
        assert f"{f16_kv * 34 / 64:>8.2f}" in out
        assert f"{f16_kv:>8.2f}" not in out

    def test_sessions_multiply_kv_and_show_llama_cpp_hint(self, tmp_path, capsys):
        calculate_vram_matrix(cache_file=write_cache(tmp_path, M=llm()), contexts=[8192], sessions=4)
        out = strip_ansi(capsys.readouterr().out)
        assert "-c 32768 -np 4" in out
        assert f"{131072 * 8192 * 4 / GIB:>8.2f}GB" in out

    def test_pure_ssm_shows_ssm_label_and_recurrent_state(self, tmp_path, capsys):
        entry = {"type": "llm", "arch": "mamba", "n_layers": 64, "n_embd": 2560, "n_kv_heads": None,
                 "kv_bytes_per_ctx_token": 0, "recurrent_state_bytes": 512 * 1024**2, "file_size_gb": 2.0}
        calculate_vram_matrix(cache_file=write_cache(tmp_path, Mamba=entry), contexts=[8192, 65536])
        out = strip_ansi(capsys.readouterr().out)
        assert "Recurrent-State: 512.0 MB" in out
        assert out.count("SSM 0.50GB") == 2

    def test_zero_kv_heads_counts_as_ssm_like_filter_js(self, tmp_path, capsys):
        entry = llm(arch="mamba", n_kv_heads=0, kv_bytes_per_ctx_token=0)
        calculate_vram_matrix(cache_file=write_cache(tmp_path, Z=entry), contexts=[8192])
        assert "SSM 0.00GB" in strip_ansi(capsys.readouterr().out)

    def test_filter_and_fits(self, tmp_path, capsys):
        cache = write_cache(
            tmp_path,
            small=llm(name="Qwen3-4B", file_size_gb=2.0),
            big=llm(name="Qwen3-32B", file_size_gb=20.0),
            other=llm(name="Gemma", file_size_gb=2.0),
        )
        calculate_vram_matrix(cache_file=cache, name_filter="qwen", fits=12, contexts=[8192])
        out = strip_ansi(capsys.readouterr().out)
        assert "Qwen3-4B" in out
        assert "Qwen3-32B" not in out
        assert "Gemma" not in out

    def test_compact_one_row_per_model_per_context(self, tmp_path, capsys):
        cache = write_cache(tmp_path, a=llm(name="Alpha"), b=llm(name="Beta"))
        calculate_vram_matrix(cache_file=cache, contexts=[8192, 32768], output="compact")
        out = strip_ansi(capsys.readouterr().out)
        assert out.count("Alpha") == 2
        assert "Kontext 8k" in out and "Kontext 32k" in out
        assert "🤖" not in out

    def test_json_output(self, tmp_path, capsys):
        entry = llm(name="Alpha", compute_buffer_bytes=GIB // 4)
        calculate_vram_matrix(cache_file=write_cache(tmp_path, a=entry), contexts=[8192], gpus=[6, 24],
                              sessions=2, output="json")
        data = json.loads(capsys.readouterr().out)
        row = data[0]["contexts"][0]
        assert data[0]["key"] == "a"
        assert row["llama_cpp"] == {"c": 16384, "np": 2}
        assert row["total_gb"] == pytest.approx(estimate(entry, 8192, sessions=2).total_gb, abs=1e-4)
        assert set(row["fits"]) == {"6 GB", "24 GB"}


class TestMain:
    def test_options_reach_calculation(self, monkeypatch):
        calls = []
        monkeypatch.setattr(vram_calculator, "calculate_vram_matrix", lambda **kw: calls.append(kw) or 0)
        vram_calculator.main([
            "--cache-type", "q4_0", "--gpu", "12", "--gpu", "24", "--ctx", "8k,32k", "--ctx", "1M",
            "--sessions", "3", "--overhead", "0.3", "--filter", "qwen", "--fits", "24", "--json",
            "--cache", "/tmp/x.json",
        ])
        vram_calculator.main([])
        assert calls[0] == {
            "cache_type": "q4_0", "cache_file": "/tmp/x.json", "gpus": [12.0, 24.0],
            "contexts": [8192, 32768, 1048576], "sessions": 3, "overhead_gb": 0.3,
            "name_filter": "qwen", "fits": 24.0, "output": "json",
        }
        assert calls[1]["cache_type"] == "f16"
        assert calls[1]["contexts"] is None
        assert calls[1]["output"] == "matrix"

    def test_cache_env(self, monkeypatch):
        calls = []
        monkeypatch.setattr(vram_calculator, "calculate_vram_matrix", lambda **kw: calls.append(kw) or 0)
        monkeypatch.setenv("VRAM_CALC_CACHE", "/env/cache.json")
        vram_calculator.main([])
        assert calls[0]["cache_file"] == "/env/cache.json"

    def test_json_and_compact_exclusive(self):
        with pytest.raises(SystemExit):
            vram_calculator.main(["--json", "--compact"])


def js_const(name, source=None):
    source = source or FILTER_JS.read_text(encoding="utf-8")
    match = re.search(r"const\s+" + name + r"\s*=\s*([0-9.]+)\s*;", source)
    assert match, f"{name} not found in filter.js"
    return float(match.group(1))


class TestConstantsInSyncWithFilterJs:
    def test_tight_fit_ratio_matches(self):
        assert js_const("TIGHT_FIT_RATIO") == vram_calculator.TIGHT_FIT_RATIO

    def test_default_overhead_matches(self):
        assert js_const("DEFAULT_OVERHEAD_GB") == vram_calculator.DEFAULT_OVERHEAD_GB

    def test_compute_mask_bytes_match(self):
        assert js_const("COMPUTE_MASK_BYTES_PER_CTX_TOKEN") == _model.COMPUTE_MASK_BYTES_PER_CTX_TOKEN

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
