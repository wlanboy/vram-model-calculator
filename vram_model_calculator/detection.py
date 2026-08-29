"""Heuristics for detecting model capabilities (tool calls, reasoning) from
GGUF metadata, plus a debug dump of all raw fields for models where the
extraction in _model.py turns up gaps.
"""
import os
import re

from .gguf_fields import decode_bytes, field_is_string, get_str, iter_decoded_parts

METADATA_DUMP_FILE = "model-metadata.txt"

THINKING_NAME_RE = re.compile(
    r'(think(?:ing)?|qwq|deepseek[-_]?r\d+|reason(?:ing)?|logic|reflect|chain|cog)',
    re.IGNORECASE
)


def detect_mcp(reader, file_path):
    """Returns True if the model supports tool calls / MCP."""
    tmpl = get_str(reader, "tokenizer.chat_template")
    if tmpl:
        tl = tmpl.lower()
        if "tool_call" in tl or "function_call" in tl or "<|tool|>" in tl or "[tool_calls]" in tl:
            return True
    tags_field = reader.fields.get("general.tags")
    if tags_field:
        for tag in iter_decoded_parts(tags_field):
            if "tool" in tag or "function-call" in tag or "mcp" in tag:
                return True
    return False


def detect_thinking(reader, name, file_path):
    """Returns True if the model supports extended thinking/reasoning."""
    # Primary signal: chat template contains <think> token
    tmpl = get_str(reader, "tokenizer.chat_template")
    if tmpl and "<think>" in tmpl:
        return True
    # Secondary signal: general.tags contains "thinking" or "reasoning"
    tags_field = reader.fields.get("general.tags")
    if tags_field:
        for tag in iter_decoded_parts(tags_field):
            if "think" in tag or "reason" in tag:
                return True
    # Fallback: name/path heuristic
    text = (name or "") + " " + os.path.basename(file_path)
    return bool(THINKING_NAME_RE.search(text))


def dump_all_fields(reader, file_path):
    """Appends all raw GGUF fields to METADATA_DUMP_FILE for debugging."""
    skip_keys = {"tokenizer.ggml.merges", "tokenizer.ggml.tokens", "tokenizer.ggml.token_type"}
    with open(METADATA_DUMP_FILE, 'a', encoding='utf-8') as f:
        f.write(f"\n{'=' * 80}\n")
        f.write(f"File: {file_path}\n")
        f.write(f"{'=' * 80}\n")
        for key in sorted(reader.fields.keys()):
            if key in skip_keys or key == "tokenizer.chat_template":
                continue
            field = reader.fields[key]
            try:
                val = field.parts[-1]
                if field_is_string(field):
                    display = decode_bytes(val, errors='replace')
                elif hasattr(val, 'tolist'):
                    lst = val.tolist()
                    display = lst[0] if isinstance(lst, list) and len(lst) == 1 else lst
                else:
                    display = str(val)
                f.write(f"  {key}: {display}\n")
            except (AttributeError, IndexError, TypeError, ValueError, UnicodeDecodeError) as e:
                f.write(f"  {key}: <read error: {e}>\n")
