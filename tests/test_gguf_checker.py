import os
import struct

from vram_model_calculator.gguf_checker import (
    GGUF_MAGIC,
    ScanReport,
    _check_header,
    _collect_gguf_files,
    check_file,
    check_shard_group,
)


def write_header(path, magic=GGUF_MAGIC, version=3, extra=b""):
    path.write_bytes(magic + struct.pack("<I", version) + extra)


class TestCheckHeader:
    def test_unreadable_file(self, tmp_path):
        missing = tmp_path / "nope.gguf"
        err = _check_header(str(missing))
        assert err is not None
        assert "nicht lesbar" in err

    def test_too_short(self, tmp_path):
        f = tmp_path / "short.gguf"
        f.write_bytes(b"GG")
        err = _check_header(str(f))
        assert err is not None
        assert "zu kurz" in err

    def test_bad_magic(self, tmp_path):
        f = tmp_path / "bad.gguf"
        write_header(f, magic=b"NOPE")
        err = _check_header(str(f))
        assert err is not None
        assert "Ungültige Magic-Bytes" in err

    def test_unsupported_version(self, tmp_path):
        f = tmp_path / "oldver.gguf"
        write_header(f, version=99)
        err = _check_header(str(f))
        assert err is not None
        assert "Nicht unterstützte GGUF-Version" in err

    def test_valid_header_returns_none(self, tmp_path):
        f = tmp_path / "ok.gguf"
        write_header(f, version=3)
        assert _check_header(str(f)) is None

    def test_supported_version_2(self, tmp_path):
        f = tmp_path / "ok2.gguf"
        write_header(f, version=2)
        assert _check_header(str(f)) is None


class TestCheckFile:
    def test_file_not_found(self, tmp_path):
        result = check_file(str(tmp_path / "missing.gguf"))
        assert result["status"] == "corrupt"
        assert "nicht gefunden" in result["reason"]

    def test_bad_header_reported_as_corrupt(self, tmp_path):
        f = tmp_path / "bad.gguf"
        write_header(f, magic=b"NOPE")
        result = check_file(str(f))
        assert result["status"] == "corrupt"

    def test_valid_header_but_unparseable_body_is_corrupt(self, tmp_path):
        f = tmp_path / "garbage.gguf"
        write_header(f, version=3, extra=b"not real gguf metadata")
        result = check_file(str(f))
        assert result["status"] in ("corrupt", "incomplete")


class TestCheckShardGroup:
    def test_all_shards_present(self):
        paths = [f"model-0000{i}-of-00003.gguf" for i in (1, 2, 3)]
        result = check_shard_group(paths)
        assert result["status"] == "ok"
        assert result["total_shards"] == 3

    def test_missing_shard_detected(self):
        paths = ["model-00001-of-00003.gguf", "model-00003-of-00003.gguf"]
        result = check_shard_group(paths)
        assert result["status"] == "incomplete"
        assert "[2]" in result["reason"]

    def test_no_shard_paths_is_corrupt(self):
        result = check_shard_group(["model.gguf"])
        assert result["status"] == "corrupt"


class TestCollectGgufFiles:
    def test_collects_single_file(self, tmp_path):
        f = tmp_path / "model.gguf"
        f.write_bytes(b"x")
        files = _collect_gguf_files([str(f)])
        assert files == [os.path.abspath(str(f))]

    def test_ignores_non_gguf_single_file(self, tmp_path):
        f = tmp_path / "readme.txt"
        f.write_bytes(b"x")
        files = _collect_gguf_files([str(f)])
        assert files == []

    def test_walks_directory_recursively(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "a.gguf").write_bytes(b"x")
        (sub / "b.gguf").write_bytes(b"x")
        (sub / "ignore.txt").write_bytes(b"x")
        files = _collect_gguf_files([str(tmp_path)])
        assert len(files) == 2
        assert all(f.endswith(".gguf") for f in files)

    def test_missing_path_reported(self, tmp_path, capsys):
        files = _collect_gguf_files([str(tmp_path / "nope")])
        assert files == []
        assert "Pfad nicht gefunden" in capsys.readouterr().out


class TestScanReport:
    def test_partitions_by_status(self):
        report = ScanReport()
        report.add({"status": "ok"}, "a.gguf")
        report.add({"status": "incomplete", "reason": "missing bytes"}, "b.gguf")
        report.add({"status": "corrupt", "reason": "bad magic"}, "c.gguf")

        assert report.ok == ["a.gguf"]
        assert report.incomplete == [("b.gguf", "missing bytes")]
        assert report.corrupt == [("c.gguf", "bad magic")]
        assert report.has_problems is True

    def test_no_problems_when_all_ok(self):
        report = ScanReport()
        report.add({"status": "ok"}, "a.gguf")
        assert report.has_problems is False
