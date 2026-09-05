from tests.conftest import FakeField, FakeReader, str_field
from vram_model_calculator.detection import detect_mcp, detect_thinking, dump_all_fields


class TestDetectMcp:
    def test_true_when_chat_template_has_tool_call(self):
        reader = FakeReader({"tokenizer.chat_template": str_field("{{ tool_call }}")})
        assert detect_mcp(reader, "model.gguf") is True

    def test_true_when_chat_template_has_function_call(self):
        reader = FakeReader({"tokenizer.chat_template": str_field("uses function_call here")})
        assert detect_mcp(reader, "model.gguf") is True

    def test_true_when_chat_template_has_bracket_tool_tag(self):
        reader = FakeReader({"tokenizer.chat_template": str_field("<|tool|> stuff")})
        assert detect_mcp(reader, "model.gguf") is True

    def test_true_when_tags_mention_tool(self):
        reader = FakeReader({"general.tags": FakeField(parts=[str_field("tool-use").parts[0]])})
        assert detect_mcp(reader, "model.gguf") is True

    def test_false_when_nothing_present(self):
        reader = FakeReader({})
        assert detect_mcp(reader, "model.gguf") is False

    def test_false_when_chat_template_unrelated(self):
        reader = FakeReader({"tokenizer.chat_template": str_field("plain chat template")})
        assert detect_mcp(reader, "model.gguf") is False


class TestDetectThinking:
    def test_true_when_chat_template_has_think_tag(self):
        reader = FakeReader({"tokenizer.chat_template": str_field("<think>reasoning</think>")})
        assert detect_thinking(reader, "MyModel", "model.gguf") is True

    def test_true_when_tags_mention_reasoning(self):
        reader = FakeReader({"general.tags": FakeField(parts=[str_field("reasoning").parts[0]])})
        assert detect_thinking(reader, "MyModel", "model.gguf") is True

    def test_true_from_name_heuristic(self):
        reader = FakeReader({})
        assert detect_thinking(reader, "DeepSeek-R1-Distill", "model.gguf") is True

    def test_true_from_path_heuristic_when_name_missing(self):
        # only the basename of file_path is checked, not the full path
        reader = FakeReader({})
        assert detect_thinking(reader, None, "/models/QwQ-32B-GGUF/QwQ-32B-Q4_K_M.gguf") is True

    def test_false_when_no_signal(self):
        reader = FakeReader({})
        assert detect_thinking(reader, "Llama 3.1 8B Instruct", "model.gguf") is False


class TestDumpAllFields:
    def test_writes_string_and_int_fields(self, tmp_path, monkeypatch):
        import vram_model_calculator.detection as detection_mod

        dump_file = tmp_path / "model-metadata.txt"
        monkeypatch.setattr(detection_mod, "METADATA_DUMP_FILE", str(dump_file))

        reader = FakeReader({
            "general.name": str_field("MyModel"),
            "general.block_count": FakeField(parts=[32], types=[0]),
            "tokenizer.chat_template": str_field("should be skipped"),
            "tokenizer.ggml.merges": str_field("should be skipped too"),
        })

        dump_all_fields(reader, "/models/MyModel/model.gguf")

        content = dump_file.read_text(encoding="utf-8")
        assert "File: /models/MyModel/model.gguf" in content
        assert "general.name: MyModel" in content
        assert "general.block_count: 32" in content
        assert "tokenizer.chat_template" not in content
        assert "tokenizer.ggml.merges" not in content

    def test_appends_rather_than_overwrites(self, tmp_path, monkeypatch):
        import vram_model_calculator.detection as detection_mod

        dump_file = tmp_path / "model-metadata.txt"
        dump_file.write_text("existing content\n", encoding="utf-8")
        monkeypatch.setattr(detection_mod, "METADATA_DUMP_FILE", str(dump_file))

        reader = FakeReader({"general.name": str_field("MyModel")})
        dump_all_fields(reader, "model.gguf")

        content = dump_file.read_text(encoding="utf-8")
        assert content.startswith("existing content\n")
        assert "general.name: MyModel" in content

    def test_handles_read_errors_gracefully(self, tmp_path, monkeypatch):
        import vram_model_calculator.detection as detection_mod

        dump_file = tmp_path / "model-metadata.txt"
        monkeypatch.setattr(detection_mod, "METADATA_DUMP_FILE", str(dump_file))

        class Broken:
            def tobytes(self):
                raise ValueError("boom")

        reader = FakeReader({"broken.field": FakeField(parts=[Broken()])})
        dump_all_fields(reader, "model.gguf")

        content = dump_file.read_text(encoding="utf-8")
        assert "broken.field: <read error:" in content
