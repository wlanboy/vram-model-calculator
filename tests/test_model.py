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

    @pytest.mark.parametrize("arch", ["dream", "llada", "llada-moe", "rnd1"])
    def test_text_diffusion_llm_arch_raises_not_an_llm(self, patch_reader, arch):
        patch_reader["reader"] = FakeReader({
            "general.type": str_field("model"),
            "general.architecture": str_field(arch),
        })
        with pytest.raises(NotAnLLMError):
            get_model_params(f"/models/{arch}/model.gguf", file_size_bytes=1)

    def test_mla_arch_computes_combined_kv_cache_dim(self, patch_reader):
        patch_reader["reader"] = llama_reader(**{
            "general.architecture": str_field("deepseek2"),
            "deepseek2.context_length": int_field(4096),
            "deepseek2.block_count": int_field(32),
            "deepseek2.embedding_length": int_field(4096),
            "deepseek2.attention.head_count": int_field(32),
            "deepseek2.feed_forward_length": int_field(11008),
            "deepseek2.attention.head_count_kv": int_field(32),
            "deepseek2.vocab_size": int_field(32000),
            "deepseek2.attention.kv_lora_rank": int_field(512),
            "deepseek2.rope.dimension_count": int_field(64),
        })
        params = get_model_params("/models/DeepSeek/model.gguf", file_size_bytes=1)
        assert params["mla_kv_dim"] == 512 + 64

    def test_mla_arch_without_lora_rank_leaves_kv_dim_none(self, patch_reader):
        patch_reader["reader"] = llama_reader(**{
            "general.architecture": str_field("mistral4"),
            "mistral4.context_length": int_field(4096),
            "mistral4.block_count": int_field(32),
            "mistral4.embedding_length": int_field(4096),
            "mistral4.attention.head_count": int_field(32),
            "mistral4.feed_forward_length": int_field(11008),
            "mistral4.attention.head_count_kv": int_field(32),
            "mistral4.vocab_size": int_field(32000),
        })
        params = get_model_params("/models/Mistral4/model.gguf", file_size_bytes=1)
        assert params["mla_kv_dim"] is None

    def test_non_mla_arch_leaves_kv_dim_none(self, patch_reader):
        patch_reader["reader"] = llama_reader()
        params = get_model_params("/models/MyModel/model.gguf", file_size_bytes=1)
        assert params["mla_kv_dim"] is None

    def test_mla_arch_with_lora_rank_but_no_rope_dim_leaves_kv_dim_none(self, patch_reader):
        patch_reader["reader"] = llama_reader(**{
            "general.architecture": str_field("deepseek2"),
            "deepseek2.context_length": int_field(4096),
            "deepseek2.block_count": int_field(32),
            "deepseek2.embedding_length": int_field(4096),
            "deepseek2.attention.head_count": int_field(32),
            "deepseek2.feed_forward_length": int_field(11008),
            "deepseek2.attention.head_count_kv": int_field(32),
            "deepseek2.vocab_size": int_field(32000),
            "deepseek2.attention.kv_lora_rank": int_field(512),
            # deepseek2.rope.dimension_count intentionally omitted
        })
        params = get_model_params("/models/DeepSeek/model.gguf", file_size_bytes=1)
        assert params["mla_kv_dim"] is None

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

    def test_audio_only_projector_uses_clip_audio_fields(self, patch_reader):
        patch_reader["reader"] = FakeReader({
            "general.type": str_field("model"),
            "general.name": str_field("MyModel-mmproj-audio"),
            "clip.has_audio_encoder": int_field(1),
            "clip.audio.num_mel_bins": int_field(128),
            "clip.audio.embedding_length": int_field(1280),
            "clip.audio.feed_forward_length": int_field(5120),
            "clip.audio.block_count": int_field(32),
        })
        params = get_model_params("/models/MyModel/mmproj-audio-f16.gguf", file_size_bytes=1)
        assert params["type"] == MODEL_TYPE_MMPROJ
        assert params["modality"] == "audio"
        assert params["num_mel_bins"] == 128
        assert params["n_embd"] == 1280
        assert params["n_layers"] == 32
        assert "has_missing_fields" not in params

    def test_vision_projector_ignores_audio_flag_when_vision_present(self, patch_reader):
        patch_reader["reader"] = FakeReader({
            "general.type": str_field("model"),
            "general.name": str_field("MyModel-mmproj"),
            "clip.has_audio_encoder": int_field(1),
            "clip.vision.image_size": int_field(336),
            "clip.vision.embedding_length": int_field(1024),
            "clip.vision.block_count": int_field(24),
        })
        params = get_model_params("/models/MyModel/mmproj-f16.gguf", file_size_bytes=1)
        assert "modality" not in params
        assert params["image_size"] == 336
