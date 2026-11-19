import json
from pathlib import Path

import pytest

from vram_model_calculator import families, hardware
from vram_model_calculator.vram_calculator import estimate

REPO = Path(__file__).resolve().parent.parent

PAGE = """<html><body>
<h1>Test GPU (12 GB)</h1>
<script>
// [schema comment]
const MODELS = [
['Model A',4.0,'Chat','Q4_K_M',3.0,25,40.0,'41K',41,'great','RUNS GREAT','S',90],
// Ergänzt aus Benchmarks
["Model B",8.0,"Chat, Code","Q8_0",9.0,75,10.0,"128K",128,"well","RUNS WELL","A",70],
];
</script>
</body></html>
"""

OVERVIEW = """<div>
  <a class="hardware-card" href="Test-GPU/index.html">
    <div class="hardware-card-title">Test GPU</div>
    <div class="hardware-card-stats">
      <span class="hardware-card-stat">99 Modelle</span>
    </div>
  </a>
</div>
"""


@pytest.fixture
def hw_dir(tmp_path):
    root = tmp_path / "hardware"
    (root / "Test-GPU").mkdir(parents=True)
    (root / "Test-GPU" / "index.html").write_text(PAGE, encoding="utf-8")
    (root / "index.html").write_text(OVERVIEW, encoding="utf-8")
    cache = {
        "_version": 1,
        "A-GGUF/a.gguf": {"type": "llm", "name": "Model-A", "quant": "Q4_K_M", "n_layers": 1, "n_embd": 1,
                          "file_size_gb": 2.0, "kv_bytes_per_ctx_token": 1024, "n_ctx_orig": 32768},
    }
    (tmp_path / "cache.json").write_text(json.dumps(cache))
    return root


class TestHardware:
    def test_device_from_page_keeps_source_comment(self):
        device = hardware.device_from_page(PAGE)
        assert device["title"] == "Test GPU (12 GB)"
        assert device["vram_gb"] == 12
        assert [m["name"] for m in device["models"]] == ["Model A", "Model B"]
        assert "source" not in device["models"][0]
        assert device["models"][1]["source"] == "Ergänzt aus Benchmarks"

    def test_match_by_normalized_name_and_quant(self):
        cache = {"k": {"name": "Model-A", "quant": "Q4_K_M"}, "j": {"name": "Model-A", "quant": "Q8_0"}}
        assert hardware.match_cache_entry({"name": "model a", "quant": "Q4_K_M"}, cache) is cache["k"]
        assert hardware.match_cache_entry({"name": "model a", "quant": "IQ2"}, cache) is None
        assert hardware.match_cache_entry({"name": "x", "quant": "y", "cache_key": "j"}, cache) is cache["j"]

    def test_generate_init_then_check(self, hw_dir, tmp_path):
        cache_file = str(tmp_path / "cache.json")
        assert hardware.generate(str(hw_dir), cache_file, check=True) == []  # no device.json, no --init
        changed = hardware.generate(str(hw_dir), cache_file, init=True)
        assert str(hw_dir / "Test-GPU" / "device.json") in changed

        page = (hw_dir / "Test-GPU" / "index.html").read_text(encoding="utf-8")
        entry = json.loads((tmp_path / "cache.json").read_text())["A-GGUF/a.gguf"]
        vram = round(estimate(entry, 8192).total_gb, 1)
        assert json.dumps(["Model A", 4.0, "Chat", "Q4_K_M", vram, round(vram / 12 * 100), 40.0, "33K", 33,
                           "great", "RUNS GREAT", "S", 90]) in page
        assert '"Model B", 8.0, "Chat, Code", "Q8_0", 9.0, 75,' in page  # unmatched: kept as stored

        overview = (hw_dir / "index.html").read_text(encoding="utf-8")
        assert "2 Modelle" in overview and "1 exzellent" in overview and "1 gut" in overview

        assert hardware.generate(str(hw_dir), cache_file, check=True) == []

    def test_check_reports_stale_page_without_writing(self, hw_dir, tmp_path):
        cache_file = str(tmp_path / "cache.json")
        hardware.generate(str(hw_dir), cache_file, init=True)
        device_path = hw_dir / "Test-GPU" / "device.json"
        device = json.loads(device_path.read_text())
        device["models"][1]["speed_ts"] = 12.5
        device_path.write_text(json.dumps(device))
        before = (hw_dir / "Test-GPU" / "index.html").read_text()
        assert hardware.generate(str(hw_dir), cache_file, check=True) == [str(hw_dir / "Test-GPU" / "index.html")]
        assert (hw_dir / "Test-GPU" / "index.html").read_text() == before


class TestFamilies:
    def test_label_from_meta_or_title(self):
        assert families.family_label('<meta name="nav-label" content="Qwen &amp; Co">', "x") == "Qwen & Co"
        assert families.family_label("<title>GLM – Modell-Übersicht</title>", "x") == "GLM"
        assert families.family_label("", "fallback") == "fallback"

    def test_generate_rewrites_marked_block(self, tmp_path):
        (tmp_path / "modell-familien").mkdir()
        (tmp_path / "modell-familien" / "b.html").write_text("<title>Beta – X</title>")
        (tmp_path / "modell-familien" / "a.html").write_text('<meta name="nav-label" content="Alpha">')
        (tmp_path / "index.html").write_text(
            "<nav>\n    <!-- BEGIN FAMILIES -->\n    <a>alt</a>\n    <!-- END FAMILIES -->\n</nav>"
        )
        assert families.generate(str(tmp_path)) is True
        html = (tmp_path / "index.html").read_text()
        assert html.index("Alpha") < html.index("Beta")
        assert "alt" not in html
        assert families.generate(str(tmp_path), check=True) is False

    def test_missing_markers(self, tmp_path):
        (tmp_path / "modell-familien").mkdir()
        (tmp_path / "index.html").write_text("<nav></nav>")
        with pytest.raises(ValueError, match="Marker"):
            families.generate(str(tmp_path))

    def test_repo_navigation_is_up_to_date(self):
        assert families.generate(str(REPO), check=True) is False, "uv run main.py families ausführen"
