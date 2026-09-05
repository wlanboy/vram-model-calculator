import pytest

from vram_model_calculator.name_utils import (
    clean_name,
    is_unreliable_name,
    name_from_path,
    resolve_name,
)


class TestCleanName:
    def test_none_and_empty_pass_through(self):
        assert clean_name(None) is None
        assert clean_name("") == ""

    @pytest.mark.parametrize("raw, expected", [
        ("allenai_olmOCR", "olmOCR"),
        ("Ibm Granite_Granite 4.0", "Granite 4.0"),
        ("Zai org_GLM", "GLM"),
    ])
    def test_strips_org_prefix(self, raw, expected):
        assert clean_name(raw) == expected

    def test_does_not_strip_when_prefix_has_digits(self):
        # prefix "70b" is not alpha, so nothing should be stripped
        assert clean_name("70b_model") == "70b_model"

    def test_does_not_strip_when_prefix_too_short(self):
        assert clean_name("a_model") == "a_model"

    def test_does_not_strip_when_prefix_too_long(self):
        prefix = "x" * 26
        name = f"{prefix}_model"
        assert clean_name(name) == name

    def test_does_not_strip_when_no_rest(self):
        assert clean_name("prefix_") == "prefix_"

    @pytest.mark.parametrize("suffix", ["GGUF", "gguf", "AWQ", "GPTQ", "EXL2", "MLX"])
    def test_strips_format_suffix(self, suffix):
        assert clean_name(f"MyModel-{suffix}") == "MyModel"
        assert clean_name(f"MyModel {suffix}") == "MyModel"

    def test_underscore_before_suffix_is_consumed_by_prefix_stripping(self):
        # A single underscore is first tried as an org-prefix separator, so
        # "MyModel_GGUF" is read as prefix="MyModel", rest="GGUF" and the
        # suffix-stripping pass never runs (nothing precedes "GGUF" anymore).
        assert clean_name("MyModel_GGUF") == "GGUF"

    def test_underscore_separator_works_when_org_prefix_precedes_it(self):
        # With a genuine org prefix present, the leftover "_GGUF" suffix is
        # still stripped normally on the second pass.
        assert clean_name("org_MyModel_GGUF") == "MyModel"

    @pytest.mark.parametrize("quant", ["BF16", "F16", "F32", "IQ4_XS", "Q4_K_M", "Q8_0"])
    def test_strips_quant_suffix(self, quant):
        assert clean_name(f"MyModel {quant}") == "MyModel"

    def test_combined_prefix_and_suffix(self):
        assert clean_name("org_MyModel-GGUF") == "MyModel"

    def test_leaves_clean_name_untouched(self):
        assert clean_name("Llama 3.1 8B Instruct") == "Llama 3.1 8B Instruct"

    def test_strips_surrounding_whitespace(self):
        assert clean_name("  MyModel  ") == "MyModel"


class TestIsUnreliableName:
    @pytest.mark.parametrize("name", [None, "", "a", "ab"])
    def test_missing_or_too_short(self, name):
        assert is_unreliable_name(name) is True

    def test_bare_hex_hash_is_unreliable(self):
        assert is_unreliable_name("deadbeefcafebabe1234") is True

    def test_short_hex_is_not_flagged_by_hash_rule(self):
        # under 16 chars, so the hex-hash branch doesn't fire and len > 2
        assert is_unreliable_name("dead") is False

    def test_normal_name_is_reliable(self):
        assert is_unreliable_name("Llama 3.1 8B Instruct") is False

    def test_hex_like_but_with_non_hex_chars_is_reliable(self):
        assert is_unreliable_name("not-a-hex-hash-name-at-all") is False


class TestNameFromPath:
    def test_uses_parent_directory(self):
        assert name_from_path("/models/MyModel-GGUF/model-Q4_K_M.gguf") == "MyModel"

    def test_strips_gguf_suffix_from_dir(self):
        assert name_from_path("/models/MyModel_GGUF/file.gguf") == "MyModel"

    def test_strips_quant_tail_from_dir(self):
        assert name_from_path("/models/MyModel-Q4_K_M/file.gguf") == "MyModel"
        assert name_from_path("/models/MyModel-IQ4_XS/file.gguf") == "MyModel"
        assert name_from_path("/models/MyModel-F16/file.gguf") == "MyModel"

    def test_single_component_path_uses_filename_stem(self):
        assert name_from_path("model.gguf") == "model"

    def test_empty_candidate_returns_none(self):
        assert name_from_path("/-GGUF/file.gguf") is None


class TestResolveName:
    def test_uses_raw_name_when_reliable(self):
        assert resolve_name("Llama 3.1 8B", "/models/whatever/file.gguf") == "Llama 3.1 8B"

    def test_falls_back_to_path_when_raw_missing(self):
        assert resolve_name(None, "/models/MyModel-GGUF/file.gguf") == "MyModel"

    def test_falls_back_to_path_when_raw_is_hash(self):
        assert resolve_name("deadbeefcafebabe1234", "/models/MyModel-GGUF/file.gguf") == "MyModel"

    def test_falls_back_to_path_when_raw_too_short(self):
        assert resolve_name("ab", "/models/MyModel-GGUF/file.gguf") == "MyModel"
