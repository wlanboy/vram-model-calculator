import json

import pytest

from vram_model_calculator import gguf_scanner
from vram_model_calculator.gguf_scanner import (
    _migrate_cache,
    _migrate_key,
    _refresh_names,
    get_shard_info,
    needs_scan,
    update_cache,
)


class TestGetShardInfo:
    def test_matches_shard_pattern(self):
        assert get_shard_info("model-00001-of-00004.gguf") == (1, 4)

    def test_case_insensitive(self):
        assert get_shard_info("model-00002-OF-00004.GGUF") == (2, 4)

    def test_non_shard_file_returns_none(self):
        assert get_shard_info("model.gguf") is None

    def test_matches_only_at_end_of_name(self):
        assert get_shard_info("model-00001-of-00004.gguf.part") is None


class TestMigrateKey:
    def test_strips_org_prefix(self):
        assert _migrate_key("SomeOrg/MyModel-GGUF/file.gguf") == "MyModel-GGUF/file.gguf"

    def test_leaves_already_relative_key_alone(self):
        assert _migrate_key("MyModel-GGUF/file.gguf") == "MyModel-GGUF/file.gguf"

    def test_single_component_key_untouched(self):
        assert _migrate_key("file.gguf") == "file.gguf"


class TestMigrateCache:
    def test_migrates_and_tags_rel_path(self):
        cache = {"SomeOrg/MyModel-GGUF/file.gguf": {"name": "MyModel"}}
        migrated = _migrate_cache(cache)
        assert "MyModel-GGUF/file.gguf" in migrated
        assert migrated["MyModel-GGUF/file.gguf"]["rel_path"] == "MyModel-GGUF/file.gguf"

    def test_deduplicates_after_migration(self, capsys):
        cache = {
            "OrgA/MyModel-GGUF/file.gguf": {"name": "A"},
            "OrgB/MyModel-GGUF/file.gguf": {"name": "B"},
        }
        migrated = _migrate_cache(cache)
        assert len(migrated) == 1

    def test_no_change_when_nothing_to_migrate(self):
        cache = {"MyModel-GGUF/file.gguf": {"name": "MyModel"}}
        migrated = _migrate_cache(cache)
        assert migrated == cache


class TestRefreshNames:
    def test_reapplies_clean_name(self):
        cache = {"k": {"name": "org_MyModel-GGUF"}}
        refreshed = _refresh_names(cache)
        assert refreshed["k"]["name"] == "MyModel"

    def test_skips_entries_without_name(self):
        cache = {"k": {"type": "adapter"}}
        refreshed = _refresh_names(cache)
        assert refreshed == cache

    def test_skips_non_dict_entries(self):
        cache = {"_version": 3}
        refreshed = _refresh_names(cache)
        assert refreshed == cache


class TestNeedsScan:
    def test_true_when_key_absent(self, tmp_path):
        f = tmp_path / "model.gguf"
        f.write_bytes(b"x" * 10)
        assert needs_scan("missing-key", str(f), {}) is True

    def test_true_when_size_changed(self, tmp_path):
        f = tmp_path / "model.gguf"
        f.write_bytes(b"x" * 10)
        cache = {"key": {"file_size_bytes": 5}}
        assert needs_scan("key", str(f), cache) is True

    def test_false_when_up_to_date(self, tmp_path):
        f = tmp_path / "model.gguf"
        f.write_bytes(b"x" * 10)
        cache = {"key": {"file_size_bytes": 10}}
        assert needs_scan("key", str(f), cache) is False

    def test_true_when_missing_fields_flag_set(self, tmp_path):
        f = tmp_path / "model.gguf"
        f.write_bytes(b"x" * 10)
        cache = {"key": {"file_size_bytes": 10, "has_missing_fields": True}}
        assert needs_scan("key", str(f), cache) is True


class TestUpdateCache:
    def test_scans_new_files_and_writes_cache(self, tmp_path, monkeypatch, capsys):
        model_dir = tmp_path / "models" / "MyModel-GGUF"
        model_dir.mkdir(parents=True)
        gguf_file = model_dir / "model-Q4_K_M.gguf"
        gguf_file.write_bytes(b"fake gguf content")

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(gguf_scanner, "CACHE_FILE", "models_cache.json")

        def fake_get_model_params(path, file_size_bytes=None):
            return {"type": "llm", "name": "MyModel", "n_layers": 32}

        monkeypatch.setattr(gguf_scanner, "get_model_params", fake_get_model_params)

        cache = update_cache([str(tmp_path / "models")])

        assert "MyModel-GGUF/model-Q4_K_M.gguf" in cache
        with open("models_cache.json") as fh:
            saved = json.load(fh)
        assert saved["_version"] == 1
        assert "MyModel-GGUF/model-Q4_K_M.gguf" in saved

    def test_skips_files_matching_skip_substrings(self, tmp_path, monkeypatch):
        model_dir = tmp_path / "models" / "heretic-model"
        model_dir.mkdir(parents=True)
        (model_dir / "model.gguf").write_bytes(b"data")

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(gguf_scanner, "CACHE_FILE", "models_cache.json")
        monkeypatch.setattr(
            gguf_scanner, "get_model_params",
            lambda *a, **kw: pytest.fail("should not be called"),
        )

        cache = update_cache([str(tmp_path / "models")])
        assert cache == {}

    def test_missing_base_dir_is_skipped_gracefully(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(gguf_scanner, "CACHE_FILE", "models_cache.json")
        cache = update_cache([str(tmp_path / "does-not-exist")])
        assert cache == {}
        assert "Pfad nicht gefunden" in capsys.readouterr().out

    def test_not_an_llm_error_is_collected_as_skip(self, tmp_path, monkeypatch, capsys):
        from vram_model_calculator._model import NotAnLLMError

        model_dir = tmp_path / "models" / "Flux-GGUF"
        model_dir.mkdir(parents=True)
        (model_dir / "model.gguf").write_bytes(b"data")

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(gguf_scanner, "CACHE_FILE", "models_cache.json")

        def raise_not_llm(path, file_size_bytes=None):
            raise NotAnLLMError("diffusion model")

        monkeypatch.setattr(gguf_scanner, "get_model_params", raise_not_llm)

        cache = update_cache([str(tmp_path / "models")])
        assert cache == {}
        assert "kein LLM" in capsys.readouterr().out
