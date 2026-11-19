import os

from vram_model_calculator.files import collect_gguf_files, group_shards


class TestCollectGgufFiles:
    def test_collects_single_file_with_its_dir_as_base(self, tmp_path):
        f = tmp_path / "model.gguf"
        f.write_bytes(b"x")
        pairs, skipped = collect_gguf_files([str(f)])
        assert pairs == [(os.path.abspath(str(f)), str(tmp_path))]
        assert skipped == 0

    def test_ignores_non_gguf_single_file(self, tmp_path):
        f = tmp_path / "readme.txt"
        f.write_bytes(b"x")
        assert collect_gguf_files([str(f)]) == ([], 0)

    def test_walks_directory_recursively(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "a.gguf").write_bytes(b"x")
        (sub / "b.gguf").write_bytes(b"x")
        (sub / "ignore.txt").write_bytes(b"x")
        pairs, _ = collect_gguf_files([str(tmp_path)])
        assert sorted(os.path.basename(p) for p, _ in pairs) == ["a.gguf", "b.gguf"]
        assert all(base == str(tmp_path) for _, base in pairs)

    def test_uppercase_extension_is_found(self, tmp_path):
        (tmp_path / "MODEL.GGUF").write_bytes(b"x")
        pairs, _ = collect_gguf_files([str(tmp_path)])
        assert len(pairs) == 1

    def test_missing_path_reported(self, tmp_path, capsys):
        assert collect_gguf_files([str(tmp_path / "nope")]) == ([], 0)
        assert "Pfad nicht gefunden" in capsys.readouterr().out

    def test_skip_patterns_match_below_base_dir_only(self, tmp_path):
        base = tmp_path / "dolphin"
        (base / "ok").mkdir(parents=True)
        (base / "ok" / "m.gguf").write_bytes(b"x")
        (base / "Heretic-7B").mkdir()
        (base / "Heretic-7B" / "m.gguf").write_bytes(b"x")
        pairs, skipped = collect_gguf_files([str(base)], ["dolphin", "heretic"])
        assert [os.path.relpath(p, base) for p, _ in pairs] == ["ok/m.gguf"]
        assert skipped == 1


class TestGroupShards:
    def test_splits_singles_and_sorts_shards(self):
        paths = ["/m/a-00002-of-00002.gguf", "/m/b.gguf", "/m/a-00001-of-00002.gguf"]
        singles, groups = group_shards(paths)
        assert singles == ["/m/b.gguf"]
        assert groups == {"/m/a.gguf": ["/m/a-00001-of-00002.gguf", "/m/a-00002-of-00002.gguf"]}
