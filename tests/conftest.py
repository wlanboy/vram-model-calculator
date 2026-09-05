"""Shared fakes for GGUFReader-based tests.

The `gguf_fields`/`detection`/`_model` modules only ever touch a small
surface of GGUFReader: `.fields` (a dict of key -> field) and, for
`gguf_fields.field_is_string`, `field.types[0]`. FakeField/FakeReader below
model just that surface without needing real GGUF binary files.
"""
import pytest

from vram_model_calculator.gguf_fields import _STRING_TYPE


class FakeField:
    """Stand-in for gguf.gguf_reader.ReaderField.

    `parts[-1]` is what the field readers actually consume: a plain str for
    string fields (get_str/decode paths just need `.tobytes()` OR fall back
    to `str(val)`), or a plain int for numeric fields (int() is called on it
    directly when it isn't numpy-array-like).
    """

    def __init__(self, parts, types=None, data=None):
        self.parts = parts
        self.types = types if types is not None else [_STRING_TYPE]
        self.data = data if data is not None else parts


class FakeBytesPart:
    """A field part that behaves like the numpy byte-array GGUF actually stores strings as."""

    def __init__(self, text):
        self._raw = text.encode("utf-8")

    def tobytes(self):
        return self._raw


class FakeReader:
    def __init__(self, fields=None):
        self.fields = fields or {}


def str_field(text):
    return FakeField(parts=[FakeBytesPart(text)], types=[_STRING_TYPE])


def int_field(value):
    return FakeField(parts=[value], types=[0])


@pytest.fixture
def make_reader():
    return lambda fields: FakeReader(fields)
