import json

import pytest

from vram_model_calculator import gguf_scanner
from vram_model_calculator.gguf_scanner import (
    _backfill_kv_bytes,
    _refresh_names,
    _short_key,
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


class TestShortKey:
    def test_strips_org_prefix(self):
        assert _short_key("SomeOrg/MyModel-GGUF/file.gguf") == "MyModel-GGUF/file.gguf"

    def test_leaves_gguf_model_folder_alone(self):
        assert _short_key("MyModel-GGUF/file.gguf") == "MyModel-GGUF/file.gguf"

    def test_single_component_key_untouched(self):
        assert _short_key("file.gguf") == "file.gguf"

    def test_hf_repo_folder_is_kept(self):
        key = "models--org--Repo-GGUF/snapshots/abc/file.gguf"
        assert _short_key(key) == key


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


class TestBackfillKvBytes:
    def test_computes_missing_field_from_stored_dimensions(self):
        cache = {
            "k": {
                "type": "llm", "n_layers": 32, "n_embd": 4096,
                "n_heads": 32, "n_kv_heads": 8,
            },
        }
        backfilled = _backfill_kv_bytes(cache)
        assert backfilled["k"]["kv_bytes_per_ctx_token"] == 2 * 32 * (8 * (4096 // 32)) * 2

    def test_leaves_existing_field_untouched(self):
        cache = {"k": {"type": "llm", "n_layers": 32, "n_embd": 4096, "kv_bytes_per_ctx_token": 999}}
        backfilled = _backfill_kv_bytes(cache)
        assert backfilled["k"]["kv_bytes_per_ctx_token"] == 999

    def test_skips_non_llm_and_incomplete_entries(self):
        cache = {
            "adapter": {"type": "adapter", "name": "lora"},
            "no-dims": {"type": "llm", "n_layers": 32},
        }
        backfilled = _backfill_kv_bytes(cache)
        assert "kv_bytes_per_ctx_token" not in backfilled["adapter"]
        assert "kv_bytes_per_ctx_token" not in backfilled["no-dims"]


class TestNeedsScan:
    def test_true_when_key_absent(self):
        assert needs_scan("missing-key", 10, {}) is True

    def test_true_when_size_changed(self):
        cache = {"key": {"file_size_bytes": 5}}
        assert needs_scan("key", 10, cache) is True

    def test_false_when_up_to_date(self):
        cache = {"key": {"file_size_bytes": 10}}
        assert needs_scan("key", 10, cache) is False

    def test_true_when_missing_fields_flag_set(self):
        cache = {"key": {"file_size_bytes": 10, "has_missing_fields": True}}
        assert needs_scan("key", 10, cache) is True

    def test_true_for_llm_entry_scanned_before_swa_support(self):
        cache = {"key": {"file_size_bytes": 10, "type": "llm"}}
        assert needs_scan("key", 10, cache) is True

    def test_false_for_llm_entry_with_swa_window_already_scanned(self):
        cache = {"key": {"file_size_bytes": 10, "type": "llm", "swa_window": None}}
        assert needs_scan("key", 10, cache) is False

    def test_false_for_non_llm_entry_without_swa_window(self):
        cache = {"key": {"file_size_bytes": 10, "type": "adapter"}}
        assert needs_scan("key", 10, cache) is False


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

    def test_up_to_date_cache_is_not_rewritten(self, tmp_path, monkeypatch):
        model_dir = tmp_path / "models" / "MyModel-GGUF"
        model_dir.mkdir(parents=True)
        (model_dir / "model.gguf").write_bytes(b"data")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(gguf_scanner, "CACHE_FILE", "models_cache.json")
        monkeypatch.setattr(gguf_scanner, "get_model_params", fake_params)

        update_cache([str(tmp_path / "models")])
        update_cache([str(tmp_path / "models")])

        with open("models_cache.json") as fh:
            assert json.load(fh)["_version"] == 1


def fake_params(path, file_size_bytes=None):
    return {"type": "llm", "name": "x", "file_size_bytes": file_size_bytes, "swa_window": None}


@pytest.fixture
def scan_env(tmp_path, monkeypatch):
    """chdir into tmp_path and record get_model_params calls."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gguf_scanner, "CACHE_FILE", "models_cache.json")
    calls = []

    def recording_params(path, file_size_bytes=None):
        calls.append((path, file_size_bytes))
        return fake_params(path, file_size_bytes)

    monkeypatch.setattr(gguf_scanner, "get_model_params", recording_params)
    return calls


class TestShards:
    def test_complete_shard_set_scanned_once_with_total_size(self, tmp_path, scan_env):
        d = tmp_path / "models" / "Big-GGUF"
        d.mkdir(parents=True)
        (d / "big-00001-of-00002.gguf").write_bytes(b"x" * 100)
        (d / "big-00002-of-00002.gguf").write_bytes(b"x" * 50)

        cache = update_cache([str(tmp_path / "models")])
        update_cache([str(tmp_path / "models")])

        assert len(scan_env) == 1
        entry = cache["Big-GGUF/big-00001-of-00002.gguf"]
        assert entry["file_size_bytes"] == 150
        assert entry["num_shards"] == 2

    def test_incomplete_shard_set_is_reported_not_scanned(self, tmp_path, scan_env, capsys):
        d = tmp_path / "models" / "Big-GGUF"
        d.mkdir(parents=True)
        (d / "big-00001-of-00003.gguf").write_bytes(b"x")
        (d / "big-00003-of-00003.gguf").write_bytes(b"x")

        cache = update_cache([str(tmp_path / "models")])

        assert scan_env == []
        assert cache == {}
        assert "unvollständiger Shard-Satz" in capsys.readouterr().out


class TestKeys:
    def test_colliding_short_keys_fall_back_to_full_path(self, tmp_path, scan_env):
        for org in ("foo", "bar"):
            d = tmp_path / "models" / org
            d.mkdir(parents=True)
            (d / "model.gguf").write_bytes(org.encode())

        cache = update_cache([str(tmp_path / "models")])

        assert set(cache) == {"foo/model.gguf", "bar/model.gguf"}

    def test_same_file_under_two_base_dirs_is_scanned_once(self, tmp_path, scan_env, capsys):
        for base in ("a", "b"):
            d = tmp_path / base / "org" / "My-GGUF"
            d.mkdir(parents=True)
            (d / "m.gguf").write_bytes(b"data")

        cache = update_cache([str(tmp_path / "a"), str(tmp_path / "b")])

        assert list(cache) == ["My-GGUF/m.gguf"]
        assert len(scan_env) == 1
        assert "doppelte Datei" in capsys.readouterr().out


class TestRobustness:
    def test_broken_symlink_does_not_abort_scan(self, tmp_path, scan_env, capsys):
        d = tmp_path / "models" / "My-GGUF"
        d.mkdir(parents=True)
        (d / "good.gguf").write_bytes(b"data")
        (d / "broken.gguf").symlink_to(tmp_path / "missing-blob")

        cache = update_cache([str(tmp_path / "models")])

        assert "My-GGUF/good.gguf" in cache
        assert "SCAN-FEHLER" in capsys.readouterr().out

    def test_non_dict_cache_entry_is_rescanned(self):
        assert needs_scan("key", 10, {"key": "garbage"}) is True

    def test_uppercase_extension_is_found(self, tmp_path, scan_env):
        d = tmp_path / "models" / "My-GGUF"
        d.mkdir(parents=True)
        (d / "MODEL.GGUF").write_bytes(b"data")

        assert "My-GGUF/MODEL.GGUF" in update_cache([str(tmp_path / "models")])

    def test_skip_filter_ignores_base_dir_path(self, tmp_path, scan_env):
        d = tmp_path / "dolphin" / "My-GGUF"
        d.mkdir(parents=True)
        (d / "m.gguf").write_bytes(b"data")

        assert "My-GGUF/m.gguf" in update_cache([str(tmp_path / "dolphin")])

    def test_no_tmp_file_left_after_save(self, tmp_path, scan_env):
        d = tmp_path / "models" / "My-GGUF"
        d.mkdir(parents=True)
        (d / "m.gguf").write_bytes(b"data")

        update_cache([str(tmp_path / "models")])

        assert not (tmp_path / "models_cache.json.tmp").exists()


class TestCachePersistence:
    def test_refreshed_cache_is_saved_without_new_scans(self, tmp_path, scan_env):
        d = tmp_path / "models" / "My-GGUF"
        d.mkdir(parents=True)
        (d / "m.gguf").write_bytes(b"data")
        old = {"_version": 3, "My-GGUF/m.gguf": {**fake_params(None, 4), "name": "org_MyModel-GGUF"}}
        (tmp_path / "models_cache.json").write_text(json.dumps(old))

        update_cache([str(tmp_path / "models")])

        assert scan_env == []
        saved = json.loads((tmp_path / "models_cache.json").read_text())
        assert saved["_version"] == 4
        assert saved["My-GGUF/m.gguf"]["name"] == "MyModel"

    def test_nested_non_gguf_folder_is_not_rescanned(self, tmp_path, scan_env):
        d = tmp_path / "models" / "org" / "humanizer"
        d.mkdir(parents=True)
        (d / "h.gguf").write_bytes(b"data")

        update_cache([str(tmp_path / "models")])
        cache = update_cache([str(tmp_path / "models")])

        assert len(scan_env) == 1
        assert list(cache) == ["humanizer/h.gguf"]


class TestPrune:
    def _setup(self, tmp_path):
        d = tmp_path / "models" / "My-GGUF"
        d.mkdir(parents=True)
        (d / "m.gguf").write_bytes(b"data")
        cache = {"_version": 1, "My-GGUF/m.gguf": fake_params(None, 4), "Gone-GGUF/x.gguf": fake_params(None, 9)}
        (tmp_path / "models_cache.json").write_text(json.dumps(cache))

    def test_stale_entries_kept_without_prune(self, tmp_path, scan_env, capsys):
        self._setup(tmp_path)
        cache = update_cache([str(tmp_path / "models")])
        assert "Gone-GGUF/x.gguf" in cache
        assert "--prune" in capsys.readouterr().out

    def test_stale_entries_removed_with_prune(self, tmp_path, scan_env):
        self._setup(tmp_path)
        cache = update_cache([str(tmp_path / "models")], prune=True)
        assert set(cache) == {"My-GGUF/m.gguf"}
        saved = json.loads((tmp_path / "models_cache.json").read_text())
        assert "Gone-GGUF/x.gguf" not in saved

    def test_prune_skipped_when_no_files_found(self, tmp_path, scan_env):
        self._setup(tmp_path)
        cache = update_cache([str(tmp_path / "empty-or-unmounted")], prune=True)
        assert "Gone-GGUF/x.gguf" in cache
