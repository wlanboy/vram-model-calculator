
from tests.conftest import (
    FakeBytesPart,
    FakeField,
    FakeReader,
    array_field,
    int_field,
    str_field,
)
from vram_model_calculator.gguf_fields import (
    FILE_TYPE_MAP,
    decode_bytes,
    field_is_string,
    get_int_list,
    get_nonneg_int,
    get_safe_int,
    get_sliding_window,
    get_str,
    get_vocab_size,
    iter_decoded_parts,
)


class TestDecodeBytes:
    def test_decodes_and_strips_nul_padding(self):
        assert decode_bytes(FakeBytesPart("hello\x00\x00")) == "hello"

    def test_plain_ascii(self):
        assert decode_bytes(FakeBytesPart("llama")) == "llama"


class TestFieldIsString:
    def test_true_for_string_type(self):
        assert field_is_string(str_field("x")) is True

    def test_false_for_non_string_type(self):
        assert field_is_string(int_field(1)) is False

    def test_false_when_types_missing(self):
        field = FakeField(parts=[1], types=[])
        assert field_is_string(field) is False

    def test_false_when_no_types_attr(self):
        class NoTypes:
            pass
        assert field_is_string(NoTypes()) is False


class TestIterDecodedParts:
    def test_yields_lowercased_parts(self):
        field = FakeField(parts=[FakeBytesPart("Tool"), FakeBytesPart("MCP")])
        assert list(iter_decoded_parts(field)) == ["tool", "mcp"]

    def test_skips_unparseable_parts(self):
        class Broken:
            def tobytes(self):
                raise UnicodeDecodeError("utf-8", b"", 0, 1, "bad")
        field = FakeField(parts=[Broken(), FakeBytesPart("ok")])
        assert list(iter_decoded_parts(field)) == ["ok"]


class TestGetStr:
    def test_missing_key_returns_none(self):
        reader = FakeReader({})
        assert get_str(reader, "general.name") is None

    def test_returns_decoded_string(self):
        reader = FakeReader({"general.name": str_field("MyModel")})
        assert get_str(reader, "general.name") == "MyModel"

    def test_falls_back_to_str_when_no_tobytes(self):
        reader = FakeReader({"general.name": FakeField(parts=["already-a-str"])})
        assert get_str(reader, "general.name") == "already-a-str"


class TestGetSafeInt:
    def test_returns_first_positive_match(self):
        reader = FakeReader({
            "arch.block_count": int_field(0),
            "arch.num_hidden_layers": int_field(32),
        })
        assert get_safe_int(reader, "arch.block_count", "arch.num_hidden_layers") == 32

    def test_returns_none_when_nothing_matches(self):
        reader = FakeReader({})
        assert get_safe_int(reader, "missing.key") is None

    def test_zero_is_not_accepted(self):
        reader = FakeReader({"k": int_field(0)})
        assert get_safe_int(reader, "k") is None

    def test_negative_is_not_accepted(self):
        reader = FakeReader({"k": int_field(-5)})
        assert get_safe_int(reader, "k") is None

    def test_unparseable_value_is_skipped(self):
        reader = FakeReader({
            "k1": FakeField(parts=["not-an-int"]),
            "k2": int_field(7),
        })
        assert get_safe_int(reader, "k1", "k2") == 7


class TestGetNonnegInt:
    def test_zero_is_accepted(self):
        reader = FakeReader({"k": int_field(0)})
        assert get_nonneg_int(reader, "k") == 0

    def test_missing_returns_none(self):
        reader = FakeReader({})
        assert get_nonneg_int(reader, "k") is None

    def test_positive_value(self):
        reader = FakeReader({"k": int_field(8)})
        assert get_nonneg_int(reader, "k") == 8


class TestGetVocabSize:
    def test_prefers_arch_specific_key(self):
        reader = FakeReader({"llama.vocab_size": int_field(32000)})
        assert get_vocab_size(reader, "llama") == 32000

    def test_falls_back_to_tokenizer_key(self):
        reader = FakeReader({"tokenizer.ggml.vocab_size": int_field(128000)})
        assert get_vocab_size(reader, "llama") == 128000

    def test_falls_back_to_token_list_length(self):
        reader = FakeReader({
            "tokenizer.ggml.tokens": FakeField(parts=[], data=list(range(100))),
        })
        assert get_vocab_size(reader, "llama") == 100

    def test_returns_none_when_nothing_available(self):
        reader = FakeReader({})
        assert get_vocab_size(reader, "llama") is None


def test_file_type_map_has_known_entries():
    assert FILE_TYPE_MAP[0] == "F32"
    assert FILE_TYPE_MAP[1] == "F16"
    assert 38 in FILE_TYPE_MAP


class TestGetIntList:
    def test_missing_key_returns_none(self):
        reader = FakeReader({})
        assert get_int_list(reader, "k") is None

    def test_scalar_field_returns_single_element_list(self):
        reader = FakeReader({"k": int_field(6)})
        assert get_int_list(reader, "k") == [6]

    def test_array_field_returns_all_elements(self):
        reader = FakeReader({"k": array_field([1, 0, 0, 1])})
        assert get_int_list(reader, "k") == [1, 0, 0, 1]

    def test_unparseable_value_returns_none(self):
        reader = FakeReader({"k": FakeField(parts=["not-an-int"])})
        assert get_int_list(reader, "k") is None


class TestGetSlidingWindow:
    def test_no_window_returns_none_zero(self):
        reader = FakeReader({})
        assert get_sliding_window(reader, "gemma3", 32) == (None, 0)

    def test_zero_layers_returns_none_zero(self):
        reader = FakeReader({"gemma3.attention.sliding_window": int_field(1024)})
        assert get_sliding_window(reader, "gemma3", 0) == (None, 0)

    def test_window_without_pattern_marks_every_layer_local(self):
        reader = FakeReader({"mistral.attention.sliding_window": int_field(4096)})
        assert get_sliding_window(reader, "mistral", 32) == (4096, 32)

    def test_periodic_pattern_every_6th_layer_is_global(self):
        # Gemma3-style: pattern=6 -> 1 global layer per 6, rest local.
        reader = FakeReader({
            "gemma3.attention.sliding_window": int_field(1024),
            "gemma3.attention.sliding_window_pattern": int_field(6),
        })
        window, swa_layers = get_sliding_window(reader, "gemma3", 12)
        assert window == 1024
        assert swa_layers == 10  # 2 global (layers 6, 12), 10 local

    def test_nonpositive_pattern_marks_every_layer_local(self):
        reader = FakeReader({
            "arch.attention.sliding_window": int_field(2048),
            "arch.attention.sliding_window_pattern": int_field(0),
        })
        assert get_sliding_window(reader, "arch", 8) == (2048, 8)

    def test_explicit_per_layer_array(self):
        # true (1) = local/SWA, false (0) = full/global attention.
        reader = FakeReader({
            "arch.attention.sliding_window": int_field(512),
            "arch.attention.sliding_window_pattern": array_field([1, 1, 0, 1, 1, 0]),
        })
        assert get_sliding_window(reader, "arch", 6) == (512, 4)
