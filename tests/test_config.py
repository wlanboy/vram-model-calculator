import pytest

from vram_model_calculator.config import (
    DEFAULT_CACHE_FILE,
    DEFAULT_SCAN_DIRS,
    DEFAULT_SKIP_PATTERNS,
    Config,
    load_config,
    metadata_dump_file,
    resolve_cache_file,
)


class TestLoadConfig:
    def test_missing_file_gives_defaults(self, tmp_path):
        config = load_config(str(tmp_path / "none.toml"))
        assert config.scan_dirs == DEFAULT_SCAN_DIRS
        assert config.skip_patterns == DEFAULT_SKIP_PATTERNS
        assert config.cache_file is None

    def test_reads_values(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text(
            'scan_dirs = ["/a", "~/b"]\ncopy_dest = "/dest"\nskip_patterns = []\ncache_file = "/c/cache.json"\n'
        )
        config = load_config(str(path))
        assert config.scan_dirs == ["/a", "~/b"]
        assert config.expanded_scan_dirs[1].endswith("/b")
        assert "~" not in config.expanded_scan_dirs[1]
        assert config.copy_dest == "/dest"
        assert config.skip_patterns == []
        assert config.cache_file == "/c/cache.json"

    def test_env_points_to_config_file(self, tmp_path, monkeypatch):
        path = tmp_path / "other.toml"
        path.write_text('copy_dest = "/x"\n')
        monkeypatch.setenv("VRAM_CALC_CONFIG", str(path))
        assert load_config().copy_dest == "/x"

    @pytest.mark.parametrize("content", ['scan_dirs = "/a"\n', "scan_dirs = [1]\n", "copy_dest = 3\n", "kaputt ="])
    def test_invalid_file_raises_value_error(self, tmp_path, content):
        path = tmp_path / "config.toml"
        path.write_text(content)
        with pytest.raises(ValueError, match="ungültig"):
            load_config(str(path))


class TestResolveCacheFile:
    def test_default(self):
        assert resolve_cache_file() == DEFAULT_CACHE_FILE

    def test_order_option_env_config(self, monkeypatch):
        config = Config(cache_file="/from/config.json")
        assert resolve_cache_file(None, config) == "/from/config.json"
        monkeypatch.setenv("VRAM_CALC_CACHE", "/from/env.json")
        assert resolve_cache_file(None, config) == "/from/env.json"
        assert resolve_cache_file("/from/option.json", config) == "/from/option.json"

    def test_dump_file_next_to_cache(self):
        assert metadata_dump_file("/data/models_cache.json") == "/data/model-metadata.txt"
