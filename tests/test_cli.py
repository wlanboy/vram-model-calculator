import pytest

from vram_model_calculator import cli, hf_cache_picker


class TestCli:
    @pytest.mark.parametrize("command", list(cli.COMMANDS))
    def test_every_command_has_help(self, command, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main([command, "--help"])
        assert exc.value.code == 0
        assert "usage" in capsys.readouterr().out

    def test_dispatches_to_module_run(self, monkeypatch):
        seen = []
        monkeypatch.setattr(cli.COMMANDS["calculate"][0], "run", lambda args: seen.append(args) or 7)
        # The parser binds run at build time, so rebuild after patching.
        assert cli.main(["calculate", "--sessions", "2"]) == 7
        assert seen[0].sessions == 2

    def test_invalid_config_is_reported_without_traceback(self, tmp_path, monkeypatch, capsys):
        bad = tmp_path / "config.toml"
        bad.write_text("scan_dirs = 5\n")
        monkeypatch.setenv("VRAM_CALC_CONFIG", str(bad))
        assert cli.main(["calculate", "--cache", str(tmp_path / "none.json")]) == 2
        assert "ungültig" in capsys.readouterr().out

    def test_alias_prepends_command(self, monkeypatch):
        seen = []
        monkeypatch.setattr(cli, "main", lambda argv: seen.append(argv) or 0)
        monkeypatch.setattr("sys.argv", ["gguf-scanner", "/models", "--prune"])
        alias = cli._alias("scan")
        assert alias() == 0
        assert seen == [["scan", "/models", "--prune"]]


class TestHfCopy:
    def test_dest_from_config_and_selection(self, tmp_path, monkeypatch, capsys):
        config = tmp_path / "config.toml"
        config.write_text(f'copy_dest = "{tmp_path / "dest"}"\n')
        monkeypatch.setenv("VRAM_CALC_CONFIG", str(config))
        files = [{"path": "/x/a.gguf", "org": "o", "repo": "r", "filename": "a.gguf", "size": 1}]
        monkeypatch.setattr(hf_cache_picker, "find_gguf_files", lambda: files)
        calls = []
        monkeypatch.setattr(hf_cache_picker, "copy_selected",
                            lambda f, idx, dest_root, dry_run: calls.append((idx, dest_root, dry_run)) or (1, 0))
        assert hf_cache_picker.main(["--dry-run", "all"]) == 0
        assert calls == [([1], str(tmp_path / "dest"), True)]
        assert hf_cache_picker.main(["--dest", "/other", "1"]) == 0
        assert calls[1] == ([1], "/other", False)
