import pytest

from tests.conftest import FakeReader, FakeTensor, array_field, int_field, str_field
from vram_model_calculator import _model
from vram_model_calculator._model import (
    MODEL_TYPE_ADAPTER,
    MODEL_TYPE_LLM,
    MODEL_TYPE_MMPROJ,
    ModelShape,
    NotAnLLMError,
    RecurrentShape,
    get_model_params,
    is_pure_ssm,
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
        # head_dim = 4096/32 = 128, kv_dim = 8*128 = 1024, 2 tensors * 32 layers * 1024 * 2 bytes
        assert params["kv_bytes_per_ctx_token"] == 2 * 32 * 1024 * 2
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
        assert params["kv_bytes_per_ctx_token"] == 0
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
        # MLA path: 32 layers * 576 combined dim * 2 bytes, not the naive
        # n_kv_heads*head_dim formula (which would hugely overstate it here).
        assert params["kv_bytes_per_ctx_token"] == 32 * 576 * 2

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

    def test_missing_architecture_with_sd_tensors_raises_not_an_llm(self, patch_reader):
        """stable-diffusion.cpp checkpoints like dreamshaper-xl-v2-turbo-Q8_0.gguf
        carry no metadata at all, not even general.architecture -- so the
        DIFFUSION_ARCHS check can't catch them. The tensor names must."""
        patch_reader["reader"] = FakeReader(
            fields={},
            tensors=[
                FakeTensor("conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight"),
                FakeTensor("model.diffusion_model.input_blocks.0.0.weight"),
                FakeTensor("first_stage_model.decoder.conv_in.weight"),
            ],
        )
        with pytest.raises(NotAnLLMError):
            get_model_params("/models/dreamshaper-xl-v2-turbo-Q8_0.gguf", file_size_bytes=1)

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

    def test_sliding_window_attention_splits_kv_cache(self, patch_reader):
        patch_reader["reader"] = llama_reader(**{
            "general.architecture": str_field("gemma3"),
            "gemma3.context_length": int_field(32768),
            "gemma3.block_count": int_field(6),
            "gemma3.embedding_length": int_field(4096),
            "gemma3.attention.head_count": int_field(32),
            "gemma3.feed_forward_length": int_field(11008),
            "gemma3.attention.head_count_kv": int_field(8),
            "gemma3.vocab_size": int_field(32000),
            "gemma3.attention.sliding_window": int_field(1024),
            "gemma3.attention.sliding_window_pattern": int_field(6),
        })
        params = get_model_params("/models/Gemma3/model.gguf", file_size_bytes=1)
        head_dim = 4096 // 32
        per_layer = 2 * (8 * head_dim) * 2
        # pattern=6 over 6 layers -> 1 global (layer 6), 5 local/SWA.
        assert params["swa_window"] == 1024
        assert params["swa_layers"] == 5
        assert params["kv_bytes_per_ctx_token"] == per_layer * 1
        assert params["kv_bytes_per_ctx_token_swa"] == per_layer * 5

    def test_key_value_length_drive_head_dim(self, patch_reader):
        # Qwen3-4B header values: n_embd/n_heads would give 80.
        patch_reader["reader"] = llama_reader(**{
            "general.architecture": str_field("qwen3"),
            "qwen3.block_count": int_field(36),
            "qwen3.embedding_length": int_field(2560),
            "qwen3.attention.head_count": int_field(32),
            "qwen3.attention.head_count_kv": int_field(8),
            "qwen3.attention.key_length": int_field(128),
            "qwen3.attention.value_length": int_field(128),
            "qwen3.vocab_size": int_field(151936),
        })
        params = get_model_params("/models/Qwen3/model.gguf", file_size_bytes=1)
        assert params["head_dim_k"] == params["head_dim_v"] == 128
        assert params["kv_layers"] == 36
        assert params["kv_bytes_per_ctx_token"] == 36 * 8 * (128 + 128) * 2

    def test_head_dim_falls_back_to_n_embd_over_n_heads(self, patch_reader):
        patch_reader["reader"] = llama_reader()
        params = get_model_params("/models/MyModel/model.gguf", file_size_bytes=1)
        assert params["head_dim_k"] == params["head_dim_v"] == 128
        assert params["head_dim_k_swa"] is None

    def test_gemma3_without_pattern_uses_period_six(self, patch_reader):
        # Gemma3 GGUFs carry sliding_window but no sliding_window_pattern.
        patch_reader["reader"] = llama_reader(**{
            "general.architecture": str_field("gemma3"),
            "gemma3.block_count": int_field(34),
            "gemma3.embedding_length": int_field(2560),
            "gemma3.attention.head_count": int_field(8),
            "gemma3.attention.head_count_kv": int_field(4),
            "gemma3.attention.key_length": int_field(256),
            "gemma3.attention.value_length": int_field(256),
            "gemma3.attention.sliding_window": int_field(1024),
            "gemma3.vocab_size": int_field(262208),
        })
        params = get_model_params("/models/Gemma3/model.gguf", file_size_bytes=1)
        per_layer = 4 * (256 + 256) * 2
        assert params["swa_layers"] == 29  # layers 6, 12, ..., 30 are global
        assert params["kv_bytes_per_ctx_token"] == per_layer * 5
        assert params["kv_bytes_per_ctx_token_swa"] == per_layer * 29

    def test_gemma4_per_layer_kv_heads_and_swa_head_dims(self, patch_reader):
        pattern = [1, 1, 1, 1, 1, 0] * 2
        patch_reader["reader"] = llama_reader(**{
            "general.architecture": str_field("gemma4"),
            "gemma4.block_count": int_field(12),
            "gemma4.embedding_length": int_field(5376),
            "gemma4.attention.head_count": int_field(32),
            "gemma4.attention.head_count_kv": array_field([16 if p else 4 for p in pattern]),
            "gemma4.attention.key_length": int_field(512),
            "gemma4.attention.value_length": int_field(512),
            "gemma4.attention.key_length_swa": int_field(256),
            "gemma4.attention.value_length_swa": int_field(256),
            "gemma4.attention.sliding_window": int_field(1024),
            "gemma4.attention.sliding_window_pattern": array_field(pattern),
            "gemma4.attention.shared_kv_layers": int_field(0),
            "gemma4.vocab_size": int_field(262144),
        })
        params = get_model_params("/models/Gemma4/model.gguf", file_size_bytes=1)
        assert params["n_kv_heads"] == 16
        assert params["head_dim_k_swa"] == 256
        assert params["kv_bytes_per_ctx_token"] == 2 * 4 * (512 + 512) * 2
        assert params["kv_bytes_per_ctx_token_swa"] == 10 * 16 * (256 + 256) * 2

    def test_shared_kv_layers_have_no_cache_of_their_own(self, patch_reader):
        patch_reader["reader"] = llama_reader(**{
            "general.architecture": str_field("gemma4"),
            "gemma4.block_count": int_field(6),
            "gemma4.embedding_length": int_field(2560),
            "gemma4.attention.head_count": int_field(8),
            "gemma4.attention.head_count_kv": int_field(2),
            "gemma4.attention.key_length": int_field(512),
            "gemma4.attention.value_length": int_field(512),
            "gemma4.attention.shared_kv_layers": int_field(2),
            "gemma4.vocab_size": int_field(262144),
        })
        params = get_model_params("/models/Gemma4/model.gguf", file_size_bytes=1)
        assert params["kv_layers"] == 4
        assert params["kv_bytes_per_ctx_token"] == 4 * 2 * (512 + 512) * 2

    def test_hybrid_full_attention_interval(self, patch_reader):
        # Qwen3.5-0.8B header values: every 4th of 24 layers is attention.
        patch_reader["reader"] = llama_reader(**{
            "general.architecture": str_field("qwen35"),
            "qwen35.block_count": int_field(24),
            "qwen35.embedding_length": int_field(1024),
            "qwen35.attention.head_count": int_field(8),
            "qwen35.attention.head_count_kv": int_field(2),
            "qwen35.attention.key_length": int_field(256),
            "qwen35.attention.value_length": int_field(256),
            "qwen35.full_attention_interval": int_field(4),
            "qwen35.vocab_size": int_field(248320),
        })
        params = get_model_params("/models/Qwen35/model.gguf", file_size_bytes=1)
        assert params["n_kv_heads"] == 2
        assert params["kv_layers"] == 6
        assert params["kv_bytes_per_ctx_token"] == 6 * 2 * (256 + 256) * 2

    def test_hybrid_per_layer_kv_heads_zero_marks_recurrent(self, patch_reader):
        patch_reader["reader"] = llama_reader(**{
            "general.architecture": str_field("lfm2"),
            "lfm2.block_count": int_field(6),
            "lfm2.embedding_length": int_field(2048),
            "lfm2.attention.head_count": int_field(32),
            "lfm2.attention.head_count_kv": array_field([0, 0, 8, 0, 0, 8]),
            "lfm2.vocab_size": int_field(65536),
        })
        params = get_model_params("/models/LFM2/model.gguf", file_size_bytes=1)
        assert params["n_kv_heads"] == 8
        assert params["kv_layers"] == 2
        assert params["kv_bytes_per_ctx_token"] == 2 * 8 * (64 + 64) * 2

    def test_unknown_arch_window_without_pattern_has_no_swa(self, patch_reader):
        patch_reader["reader"] = llama_reader(**{"llama.attention.sliding_window": int_field(4096)})
        params = get_model_params("/models/MyModel/model.gguf", file_size_bytes=1)
        assert params["swa_window"] is None
        assert params["kv_bytes_per_ctx_token"] == 32 * 8 * (128 + 128) * 2

    def test_no_sliding_window_leaves_new_fields_empty(self, patch_reader):
        patch_reader["reader"] = llama_reader()
        params = get_model_params("/models/MyModel/model.gguf", file_size_bytes=1)
        assert params["swa_window"] is None
        assert params["swa_layers"] == 0
        assert params["kv_bytes_per_ctx_token_swa"] == 0


class TestModelShapeKvBytesPerCtxToken:
    """Direct unit tests for the one place the KV-cache formula lives: every
    other consumer (vram_calculator.py, filter.js) just multiplies this
    precomputed value by a context length."""

    def test_classic_attention_formula(self):
        shape = ModelShape(n_layers=32, n_embd=4096, n_heads=32, n_kv_heads=8)
        head_dim = 4096 // 32
        kv_dim = 8 * head_dim
        assert shape.kv_bytes_per_ctx_token() == 2 * 32 * kv_dim * 2

    def test_mla_dim_overrides_classic_formula(self):
        shape = ModelShape(n_layers=32, n_embd=4096, n_heads=32, n_kv_heads=32, mla_kv_dim=576)
        assert shape.kv_bytes_per_ctx_token() == 32 * 576 * 2

    def test_ssm_has_no_kv_cache(self):
        shape = ModelShape(n_layers=24, n_embd=2048, n_heads=16, n_kv_heads=None)
        assert shape.kv_layers == 0
        assert shape.kv_bytes_per_ctx_token() == 0

    def test_zero_kv_heads_has_no_kv_cache(self):
        shape = ModelShape(n_layers=24, n_embd=2048, n_heads=16, n_kv_heads=0)
        assert shape.kv_bytes_per_ctx_token() == 0

    def test_swa_splits_global_and_local_contribution(self):
        shape = ModelShape(
            n_layers=6, n_embd=4096, n_heads=32, n_kv_heads=8,
            swa_pattern=(True,) * 5 + (False,), swa_window=1024,
        )
        assert shape.swa_layers == 5
        head_dim = 4096 // 32
        per_layer = 2 * (8 * head_dim) * 2
        assert shape.kv_bytes_per_ctx_token() == per_layer * 1
        assert shape.kv_bytes_per_ctx_token_swa() == per_layer * 5

    def test_swa_layers_without_window_falls_back_to_full_attention(self):
        # swa_window unset means the pattern info wasn't usable; treat every
        # layer as global rather than silently dropping the local layers' KV cache.
        shape = ModelShape(
            n_layers=6, n_embd=4096, n_heads=32, n_kv_heads=8,
            swa_pattern=(True,) * 5 + (False,), swa_window=None,
        )
        assert shape.swa_layers == 0
        head_dim = 4096 // 32
        per_layer = 2 * (8 * head_dim) * 2
        assert shape.kv_bytes_per_ctx_token() == per_layer * 6
        assert shape.kv_bytes_per_ctx_token_swa() == 0

    def test_mla_arch_ignores_swa_layers(self):
        shape = ModelShape(
            n_layers=32, n_embd=4096, n_heads=32, n_kv_heads=32, mla_kv_dim=576,
            swa_pattern=(True,) * 10 + (False,) * 22, swa_window=1024,
        )
        assert shape.kv_bytes_per_ctx_token() == 32 * 576 * 2
        assert shape.kv_bytes_per_ctx_token_swa() == 0

    def test_key_value_length_override_n_embd_over_n_heads(self):
        # Qwen3-4B: n_embd/n_heads = 80, but key_length = value_length = 128.
        shape = ModelShape(n_layers=36, n_embd=2560, n_heads=32, n_kv_heads=8, head_dim_k=128, head_dim_v=128)
        assert shape.kv_bytes_per_ctx_token() == 36 * 8 * (128 + 128) * 2

    def test_key_and_value_length_sized_separately(self):
        shape = ModelShape(n_layers=2, n_embd=4096, n_heads=32, n_kv_heads=4, head_dim_k=192, head_dim_v=128)
        assert shape.kv_bytes_per_ctx_token() == 2 * 4 * (192 + 128) * 2

    def test_swa_layers_use_their_own_head_dims_and_kv_heads(self):
        # Gemma-4-31B-style: local layers 16 KV heads x 256, global 4 x 512.
        shape = ModelShape(
            n_layers=6, n_embd=5376, n_heads=32, n_kv_heads=16,
            swa_window=1024, swa_pattern=(True,) * 5 + (False,),
            kv_heads_per_layer=(16, 16, 16, 16, 16, 4),
            head_dim_k=512, head_dim_v=512, head_dim_k_swa=256, head_dim_v_swa=256,
        )
        assert shape.kv_bytes_per_ctx_token() == 1 * 4 * (512 + 512) * 2
        assert shape.kv_bytes_per_ctx_token_swa() == 5 * 16 * (256 + 256) * 2

    def test_swa_head_dims_default_to_full_ones(self):
        shape = ModelShape(
            n_layers=2, n_embd=2560, n_heads=8, n_kv_heads=4,
            swa_window=1024, swa_pattern=(True, False), head_dim_k=256, head_dim_v=256,
        )
        assert shape.kv_bytes_per_ctx_token_swa() == 4 * 512 * 2

    def test_zero_kv_heads_in_per_layer_counts_skip_layer(self):
        shape = ModelShape(n_layers=4, n_embd=1024, n_heads=8, n_kv_heads=2, kv_heads_per_layer=(0, 0, 0, 2))
        assert shape.kv_layers == 1
        assert shape.kv_bytes_per_ctx_token() == 2 * (128 + 128) * 2


class TestIsPureSsm:
    @pytest.mark.parametrize("n_kv_heads", [None, 0])
    def test_no_kv_heads_is_pure_ssm(self, n_kv_heads):
        assert is_pure_ssm(n_kv_heads) is True

    def test_kv_heads_is_not_pure_ssm(self):
        assert is_pure_ssm(8) is False


class TestRecurrentShape:
    """Mirrors llama_hparams::n_embd_r()/n_embd_s() in llama.cpp; the state
    is f32 (4 bytes per element) per recurrent layer and session."""

    def test_no_recurrent_layers_has_no_state(self):
        shape = RecurrentShape(n_recurrent_layers=0, n_embd=4096, n_heads=32, ssm_d_conv=4, ssm_d_inner=8192)
        assert shape.state_bytes_per_session() == 0

    def test_mamba1_conv_and_ssm_state(self):
        shape = RecurrentShape(n_recurrent_layers=4, n_embd=768, n_heads=0, ssm_d_conv=4, ssm_d_inner=1536, ssm_d_state=16)
        assert shape.n_embd_r() == 3 * 1536
        assert shape.n_embd_s() == 16 * 1536
        assert shape.state_bytes_per_session() == 4 * (3 * 1536 + 16 * 1536) * 4

    def test_mamba2_groups_widen_conv_state(self):
        shape = RecurrentShape(
            n_recurrent_layers=1, n_embd=2048, n_heads=0,
            ssm_d_conv=4, ssm_d_inner=2048, ssm_d_state=128, ssm_n_group=16,
        )
        assert shape.n_embd_r() == 3 * (2048 + 2 * 16 * 128)
        assert shape.n_embd_s() == 128 * 2048

    def test_rwkv_token_shift_and_wkv_state(self):
        shape = RecurrentShape(n_recurrent_layers=1, n_embd=2048, n_heads=0, wkv_head_size=64)
        assert shape.n_embd_r() == 2 * 2048
        assert shape.n_embd_s() == 2048 * 64

    def test_lfm2_shortconv_has_no_ssm_state(self):
        shape = RecurrentShape(n_recurrent_layers=1, n_embd=2048, n_heads=32, shortconv_l_cache=3)
        assert shape.n_embd_r() == 2048 * 2
        assert shape.n_embd_s() == 0

    def test_kda_conv_for_qkv_and_square_state_per_head(self):
        shape = RecurrentShape(n_recurrent_layers=1, n_embd=2304, n_heads=32, ssm_d_conv=4, kda_head_dim=128)
        assert shape.n_embd_r() == 3 * 3 * 32 * 128
        assert shape.n_embd_s() == 128 * 128 * 32

    def test_minimax_linear_attention_state(self):
        shape = RecurrentShape(n_recurrent_layers=1, n_embd=6144, n_heads=64, la_head_dim=128)
        assert shape.n_embd_r() == 0
        assert shape.n_embd_s() == 128 * 128 * 64


class TestGetModelParamsRecurrentState:
    def test_transformer_has_no_recurrent_state(self, patch_reader):
        patch_reader["reader"] = llama_reader()
        params = get_model_params("/models/MyModel/model.gguf", file_size_bytes=1)
        assert params["recurrent_layers"] == 0
        assert params["recurrent_state_bytes"] == 0

    def test_qwen35_interval_hybrid(self, patch_reader):
        # Qwen3.5-2B header values; llama.cpp reports 19.27 MiB RS buffer per sequence.
        patch_reader["reader"] = llama_reader(**{
            "general.architecture": str_field("qwen35"),
            "qwen35.block_count": int_field(24),
            "qwen35.embedding_length": int_field(2048),
            "qwen35.attention.head_count": int_field(8),
            "qwen35.attention.head_count_kv": int_field(2),
            "qwen35.attention.key_length": int_field(256),
            "qwen35.attention.value_length": int_field(256),
            "qwen35.full_attention_interval": int_field(4),
            "qwen35.ssm.conv_kernel": int_field(4),
            "qwen35.ssm.state_size": int_field(128),
            "qwen35.ssm.group_count": int_field(16),
            "qwen35.ssm.time_step_rank": int_field(16),
            "qwen35.ssm.inner_size": int_field(2048),
            "qwen35.vocab_size": int_field(248320),
        })
        params = get_model_params("/models/Qwen35/model.gguf", file_size_bytes=1)
        assert params["recurrent_layers"] == 18
        assert params["recurrent_state_bytes"] == 18 * (3 * (2048 + 2 * 16 * 128) + 128 * 2048) * 4 == 20201472

    def test_lfm2_zero_kv_heads_are_recurrent_layers(self, patch_reader):
        patch_reader["reader"] = llama_reader(**{
            "general.architecture": str_field("lfm2"),
            "lfm2.block_count": int_field(6),
            "lfm2.embedding_length": int_field(2048),
            "lfm2.attention.head_count": int_field(32),
            "lfm2.attention.head_count_kv": array_field([0, 0, 8, 0, 0, 8]),
            "lfm2.shortconv.l_cache": int_field(3),
            "lfm2.vocab_size": int_field(65536),
        })
        params = get_model_params("/models/LFM2/model.gguf", file_size_bytes=1)
        assert params["recurrent_layers"] == 4
        assert params["recurrent_state_bytes"] == 4 * 2048 * 2 * 4

    def test_nemotron_h_mlp_layers_are_not_recurrent(self, patch_reader):
        # Layer types: M - M * M -  (Mamba, MLP, attention); MLP layers have
        # head_count_kv 0 too, but a non-zero feed_forward_length.
        patch_reader["reader"] = llama_reader(**{
            "general.architecture": str_field("nemotron_h"),
            "nemotron_h.block_count": int_field(6),
            "nemotron_h.embedding_length": int_field(3136),
            "nemotron_h.attention.head_count": int_field(40),
            "nemotron_h.attention.head_count_kv": array_field([0, 0, 0, 8, 0, 0]),
            "nemotron_h.feed_forward_length": array_field([0, 12544, 0, 0, 0, 12544]),
            "nemotron_h.ssm.conv_kernel": int_field(4),
            "nemotron_h.ssm.inner_size": int_field(4096),
            "nemotron_h.ssm.state_size": int_field(128),
            "nemotron_h.ssm.group_count": int_field(8),
            "nemotron_h.vocab_size": int_field(131072),
        })
        params = get_model_params("/models/Nemotron/model.gguf", file_size_bytes=1)
        assert params["recurrent_layers"] == 3
        assert params["recurrent_state_bytes"] == 3 * (3 * (4096 + 2 * 8 * 128) + 128 * 4096) * 4

    def test_pure_mamba_every_layer_recurrent(self, patch_reader):
        patch_reader["reader"] = llama_reader(**{
            "general.architecture": str_field("mamba"),
            "mamba.block_count": int_field(4),
            "mamba.embedding_length": int_field(768),
            "mamba.ssm.conv_kernel": int_field(4),
            "mamba.ssm.inner_size": int_field(1536),
            "mamba.ssm.state_size": int_field(16),
            "mamba.vocab_size": int_field(50280),
        })
        params = get_model_params("/models/Mamba/model.gguf", file_size_bytes=1)
        assert params["n_kv_heads"] is None
        assert params["recurrent_layers"] == 4
        assert params["recurrent_state_bytes"] == 4 * (3 * 1536 + 16 * 1536) * 4

    def test_hybrid_without_layer_info_counts_all_layers(self, patch_reader):
        # Scalar head_count_kv: the attention layers can't be told apart, so
        # the model counts as pure SSM (no KV cache) with state in every layer.
        patch_reader["reader"] = llama_reader(**{
            "general.architecture": str_field("jamba"),
            "jamba.block_count": int_field(8),
            "jamba.embedding_length": int_field(4096),
            "jamba.attention.head_count": int_field(32),
            "jamba.attention.head_count_kv": int_field(8),
            "jamba.ssm.conv_kernel": int_field(4),
            "jamba.ssm.inner_size": int_field(8192),
            "jamba.ssm.state_size": int_field(16),
            "jamba.vocab_size": int_field(65536),
        })
        params = get_model_params("/models/Jamba/model.gguf", file_size_bytes=1)
        assert params["n_kv_heads"] is None
        assert params["recurrent_layers"] == 8


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
