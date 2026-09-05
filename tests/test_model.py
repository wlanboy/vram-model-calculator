import pytest

from tests.conftest import FakeReader, int_field, str_field
from vram_model_calculator import _model
from vram_model_calculator._model import (
    MODEL_TYPE_ADAPTER,
    MODEL_TYPE_LLM,
    MODEL_TYPE_MMPROJ,
    NotAnLLMError,
    get_model_params,
)


def llama_reader(**overrides):
    fields = {
        "general.type": str_field("model"),
        "general.architecture": str_field("llama"),
        "general.name": str_field("MyModel"),
        "llama.context_length": int_field(4096),
        "general.file_type": int_field(1),
        "llama.block_count": int_field(32),
        "llama.embedding_length": int_field(4096),
        "llama.attention.head_count": int_field(32),
        "llama.feed_forward_length": int_field(11008),
        "llama.attention.head_count_kv": int_field(8),
        "llama.vocab_size": int_field(32000),
    }
    fields.update(overrides)
    return FakeReader(fields)


@pytest.fixture(autouse=True)
def patch_reader(monkeypatch, tmp_path):
    """Point open_gguf_reader at whatever reader the test stashes on the module,
    and redirect the metadata dump file so tests don't write into the repo."""
    holder = {}

    def fake_open(file_path):
        return holder["reader"]

    monkeypatch.setattr(_model, "open_gguf_reader", fake_open)
    monkeypatch.setattr(
        "vram_model_calculator.detection.METADATA_DUMP_FILE",
        str(tmp_path / "model-metadata.txt"),
    )
    return holder


class TestGetModelParamsLlm:
    def test_extracts_core_fields(self, patch_reader):
        patch_reader["reader"] = llama_reader()
        params = get_model_params("/models/MyModel/model.gguf", file_size_bytes=1024**3)

        assert params["type"] == MODEL_TYPE_LLM
        assert params["arch"] == "llama"
        assert params["name"] == "MyModel"
        assert params["n_layers"] == 32
        assert params["n_embd"] == 4096
        assert params["n_heads"] == 32
        assert params["n_kv_heads"] == 8
        assert params["n_ctx_orig"] == 4096
        assert params["quant"] == "F16"
        assert params["vocab_size"] == 32000
        assert params["file_size_gb"] == 1.0
        assert "has_missing_fields" not in params

    def test_gqa_zero_kv_heads_means_same_as_heads(self, patch_reader):
        patch_reader["reader"] = llama_reader(**{
            "llama.attention.head_count_kv": int_field(0),
        })
        params = get_model_params("/models/MyModel/model.gguf", file_size_bytes=1)
        assert params["n_kv_heads"] == params["n_heads"] == 32

    def test_ssm_arch_has_no_kv_heads(self, patch_reader):
        patch_reader["reader"] = llama_reader(**{
            "general.architecture": str_field("mamba"),
            "mamba.context_length": int_field(4096),
            "mamba.block_count": int_field(32),
            "mamba.embedding_length": int_field(4096),
            "mamba.attention.head_count": int_field(32),
            "mamba.feed_forward_length": int_field(11008),
            "mamba.vocab_size": int_field(32000),
        })
        params = get_model_params("/models/MyModel/model.gguf", file_size_bytes=1)
        assert params["n_kv_heads"] is None
        assert "has_missing_fields" not in params

    def test_diffusion_arch_raises_not_an_llm(self, patch_reader):
        patch_reader["reader"] = FakeReader({
            "general.type": str_field("model"),
            "general.architecture": str_field("flux"),
        })
        with pytest.raises(NotAnLLMError):
            get_model_params("/models/Flux/model.gguf", file_size_bytes=1)

    def test_missing_architecture_falls_back_to_llama(self, patch_reader):
        reader = llama_reader()
        del reader.fields["general.architecture"]
        patch_reader["reader"] = reader
        params = get_model_params("/models/MyModel/model.gguf", file_size_bytes=1)
        assert params["arch"] == "llama"

    def test_invalid_layer_count_raises_value_error(self, patch_reader):
        patch_reader["reader"] = llama_reader(**{"llama.block_count": int_field(0)})
        with pytest.raises(ValueError):
            get_model_params("/models/MyModel/model.gguf", file_size_bytes=1)

    def test_missing_critical_field_sets_flag(self, patch_reader):
        reader = llama_reader()
        del reader.fields["llama.vocab_size"]
        patch_reader["reader"] = reader
        params = get_model_params("/models/MyModel/model.gguf", file_size_bytes=1)
        assert params["has_missing_fields"] is True
        assert params["vocab_size"] is None

    def test_unreliable_name_falls_back_to_path(self, patch_reader):
        patch_reader["reader"] = llama_reader(**{"general.name": str_field("ab")})
        params = get_model_params("/models/CoolModel-GGUF/model.gguf", file_size_bytes=1)
        assert params["name"] == "CoolModel"

    def test_moe_expert_fields_extracted(self, patch_reader):
        patch_reader["reader"] = llama_reader(**{
            "llama.expert_count": int_field(8),
            "llama.expert_used_count": int_field(2),
        })
        params = get_model_params("/models/MyModel/model.gguf", file_size_bytes=1)
        assert params["n_experts"] == 8
        assert params["n_experts_used"] == 2


class TestGetModelParamsAdapter:
    def test_adapter_type_short_circuits(self, patch_reader):
        patch_reader["reader"] = FakeReader({
            "general.type": str_field("adapter"),
            "general.name": str_field("MyLora"),
        })
        params = get_model_params("/models/MyLora/adapter.gguf", file_size_bytes=2 * 1024**3)
        assert params == {
            "type": MODEL_TYPE_ADAPTER,
            "name": "MyLora",
            "file_size_bytes": 2 * 1024**3,
            "file_size_gb": 2.0,
        }


class TestGetModelParamsMmproj:
    def test_detected_by_filename(self, patch_reader):
        patch_reader["reader"] = FakeReader({
            "general.type": str_field("model"),
            "general.name": str_field("MyModel-mmproj"),
            "clip.vision.image_size": int_field(336),
            "clip.vision.embedding_length": int_field(1024),
            "clip.vision.block_count": int_field(24),
        })
        params = get_model_params("/models/MyModel/mmproj-f16.gguf", file_size_bytes=1)
        assert params["type"] == MODEL_TYPE_MMPROJ
        assert params["image_size"] == 336
        assert params["n_layers"] == 24

    def test_detected_by_general_type(self, patch_reader):
        patch_reader["reader"] = FakeReader({
            "general.type": str_field("projector"),
            "general.name": str_field("VisionTower"),
            "clip.vision.image_size": int_field(336),
            "clip.vision.embedding_length": int_field(1024),
            "clip.vision.block_count": int_field(24),
        })
        params = get_model_params("/models/MyModel/vision.gguf", file_size_bytes=1)
        assert params["type"] == MODEL_TYPE_MMPROJ

    def test_missing_critical_field_sets_flag(self, patch_reader):
        patch_reader["reader"] = FakeReader({
            "general.type": str_field("model"),
            "general.name": str_field("MyModel-mmproj"),
        })
        params = get_model_params("/models/MyModel/mmproj-f16.gguf", file_size_bytes=1)
        assert params["has_missing_fields"] is True
