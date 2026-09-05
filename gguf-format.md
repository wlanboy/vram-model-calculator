# GGUF-Format – Technische Übersicht

GGUF (GGML Universal Format) ist das binäre Dateiformat für KI-Modelle, das von llama.cpp verwendet wird. Es löste das ältere GGML- und GGJT-Format ab und wurde so konzipiert, dass Metadaten, Tokenizer und Tensor-Gewichte in einer einzigen Datei gespeichert werden können.

---

## Dateistruktur

Eine GGUF-Datei besteht aus drei Abschnitten:

1. **Header** – Magic Bytes (`GGUF`), Versionsnummer, Anzahl der Tensor-Einträge und Metadaten-Key-Value-Paare
2. **Metadaten** – Key-Value-Paare mit Architektur, Tokenizer, Modellparametern und weiteren Informationen
3. **Tensordaten** – Die eigentlichen Modellgewichte (ausgerichtet auf 32-Byte-Grenzen)

---

## Modelltypen (`general.type`)

Das Feld `general.type` gibt an, welchen Typ eine GGUF-Datei hat:

| Wert | Bedeutung |
|------|-----------|
| `model` | Vollständiges Sprachmodell (LLM) |
| `adapter` | LoRA- oder PEFT-Adapter (kein eigenständiges Modell) |
| `projector` | Multimodaler Projektor (z.B. CLIP-Vision für LLaVA) |

Adapter-Dateien enthalten keine vollständigen Schichten und können nicht eigenständig zur VRAM-Berechnung herangezogen werden. Projektoren (`mmproj-*.gguf`) haben eine andere Metadatenstruktur als LLMs.

---

## Allgemeine Metadatenfelder

| Feld | Typ | Beschreibung |
|------|-----|--------------|
| `general.architecture` | string | Architekturname (z.B. `llama`, `mistral`, `qwen2`) |
| `general.name` | string | Modellname |
| `general.type` | string | Dateityp: `model`, `adapter`, `projector` |
| `general.size_label` | string | Lesbare Parametergröße, z.B. `"8B"`, `"70B"`, `"8x7B"` |
| `general.parameter_count` | uint64 | Exakte Parameteranzahl (falls vorhanden) |
| `general.file_type` | uint32 | Quantisierungstyp als Enum-Wert (siehe `LlamaFileType`) |
| `general.context_length` | uint32 | Maximale Kontextlänge (Fallback falls arch-spezifisch fehlt) |
| `general.basename` | string | Modellfamilien-Name |
| `general.finetune` | string | Fine-Tuning-Variante (z.B. `Chat`, `Instruct`) |
| `general.version` | string | Modellversion |

---

## Architekturspezifische Felder

Alle architekturspezifischen Felder verwenden das Schema `{arch}.feldname`, wobei `{arch}` dem Wert von `general.architecture` entspricht.

### Basisparameter

| Feld | Typ | Beschreibung |
|------|-----|--------------|
| `{arch}.block_count` | uint32 | Anzahl der Transformer-Blöcke (Schichten) |
| `{arch}.embedding_length` | uint32 | Embedding-Dimension (`d_model`) |
| `{arch}.feed_forward_length` | uint32 | Größe des Feed-Forward-Layers |
| `{arch}.context_length` | uint32 | Maximale Kontextlänge des Modells |
| `{arch}.vocab_size` | uint32 | Vokabulargröße |

### Attention-Parameter (nur Attention-Architekturen)

| Feld | Typ | Beschreibung |
|------|-----|--------------|
| `{arch}.attention.head_count` | uint32 | Anzahl der Query-Heads |
| `{arch}.attention.head_count_kv` | uint32 | Anzahl der KV-Heads (GQA/MQA); `0` bedeutet identisch mit `head_count` |
| `{arch}.attention.key_length` | uint32 | Head-Dimension für Keys (falls nicht standard) |
| `{arch}.attention.value_length` | uint32 | Head-Dimension für Values (falls nicht standard) |

### RoPE-Parameter

| Feld | Typ | Beschreibung |
|------|-----|--------------|
| `{arch}.rope.dimension_count` | uint32 | Anzahl der RoPE-Dimensionen pro Head |
| `{arch}.rope.freq_base` | float32 | Basisfrequenz (Standard ~10.000; erweiterte Kontextmodelle nutzen bis 1.000.000) |
| `{arch}.rope.scale_linear` | float32 | Linearer Skalierungsfaktor für Kontexterweiterung |

### MoE-Parameter (Mixture of Experts)

| Feld | Typ | Beschreibung |
|------|-----|--------------|
| `{arch}.expert_count` | uint32 | Gesamtzahl der Experten |
| `{arch}.expert_used_count` | uint32 | Aktivierte Experten pro Token-Durchlauf |

---

## Architekturen

### Reine Attention-Architekturen (mit `n_kv_heads`)

Alle klassischen Transformer-Architekturen wie `llama`, `mistral`, `qwen2`, `gemma`, `gemma2`, `gemma3`, `phi`, `phi3`, `falcon`, `starcoder2`, `deepseek2`, `command-r` u.a. verwenden Attention-Layer und benötigen `attention.head_count_kv` für die KV-Cache-VRAM-Berechnung.

### SSM- und Hybrid-Architekturen (kein `n_kv_heads`)

Diese Architekturen liefern kein global auswertbares `attention.head_count_kv` — entweder weil sie komplett auf State-Space-Blöcken basieren (reine SSMs), oder weil sie SSM- und Attention-Blöcke mischen und der Wert für die Attention-Layer nicht als einzelnes globales Feld vorliegt (Hybrid-Modelle). Beide Gruppen werden vom Scanner identisch behandelt: `n_kv_heads` wird auf `None` gesetzt statt aus der GGUF-Datei gelesen.

| Architektur | Typ | Beschreibung |
|-------------|-----|--------------|
| `mamba` | SSM | Mamba (SSM v1) |
| `mamba2` | SSM | Mamba2 (State Space Model v2) |
| `rwkv` | SSM | RWKV (ältere Versionen) |
| `rwkv6` | SSM | RWKV v6 |
| `rwkv7` | SSM | RWKV v7 (Eagle/Finch) |
| `rwkv6qwen2` | Hybrid | RWKV6/Qwen2-Hybrid |
| `arwkv7` | Hybrid | ARWKV v7 |
| `jamba` | Hybrid | Jamba (AI21 Labs) |
| `falcon_h1` | Hybrid | Falcon H1 (Hybrid Head) |
| `granite_hybrid` | Hybrid | IBM Granite Hybrid |
| `plamo2` | Hybrid | PLaMo 2 |
| `plamo3` | Hybrid | PLaMo 3 |
| `qwen3next` | Hybrid | Qwen3-Next |
| `lfm2` | Hybrid | LiquidAI LFM2 |
| `lfm2moe` | Hybrid | LiquidAI LFM2 (MoE) |
| `nemotron_h` | Hybrid | Nvidia Nemotron-H |
| `nemotron_h_moe` | Hybrid | Nvidia Nemotron-H (MoE) |

Diese Liste (`SSM_ARCHS` in `_model.py`) wächst mit jeder neuen State-Space-/Hybrid-Architektur, die llama.cpp unterstützt, und muss bei Bedarf ergänzt werden.

---

## Nicht unterstützte Architekturen (Diffusionsmodelle)

Bild-/Video-Diffusionsmodelle (aus stable-diffusion.cpp-GGUF-Quantisierungen, z.B. über gemeinsam genutzte HF-Caches) haben keine LLM-typischen `block_count`/`n_layers`-Metadaten und liegen außerhalb des Scopes dieses Rechners. Sie werden anhand von `general.architecture` erkannt und beim Scannen übersprungen (`NotAnLLMError`):

| Architektur | Beschreibung |
|-------------|--------------|
| `flux` | Flux |
| `sd1`, `sd2`, `sd3` | Stable Diffusion 1/2/3 |
| `sdxl`, `sdxl_refiner` | Stable Diffusion XL |
| `chroma` | Chroma |
| `lumina2` | Lumina 2 |
| `auraflow` | AuraFlow |
| `hidream` | HiDream |
| `hunyuan_video` | Hunyuan Video |
| `wan`, `wan2` | Wan / Wan2 |
| `ltxv` | LTX-Video |
| `cosmos` | Nvidia Cosmos |
| `qwen_image` | Qwen-Image |
| `pixart` | PixArt |
| `kolors` | Kolors |
| `cascade` | Stable Cascade |
| `playground` | Playground |

---

## Quantisierungstypen (`general.file_type`)

Der Wert von `general.file_type` ist ein Integer, der über die Enum-Klasse `LlamaFileType` aus dem `gguf`-Paket aufgelöst werden kann. Wichtige Werte:

| ID | Name | Beschreibung |
|----|------|--------------|
| 0 | F32 | 32-bit Float |
| 1 | F16 | 16-bit Float |
| 28 | BF16 | Brain Float 16 |
| 2 | Q4_0 | 4-bit Quantisierung (älteres Format) |
| 15 | Q4_K_M | 4-bit K-Quant, Medium |
| 17 | Q5_K_M | 5-bit K-Quant, Medium |
| 18 | Q6_K | 6-bit K-Quant |
| 7 | Q8_0 | 8-bit Quantisierung |
| 38 | MXFP4 | Microscaling FP4 (neueres Format) |

---

## Tokenizer-Felder

| Feld | Typ | Beschreibung |
|------|-----|--------------|
| `tokenizer.ggml.model` | string | Tokenizer-Typ: `bpe`, `sentencepiece`, `llama` |
| `tokenizer.ggml.tokens` | array | Token-Vokabular (Anzahl = `vocab_size`) |
| `tokenizer.ggml.token_type` | array | Token-Typ-Flags |
| `tokenizer.ggml.merges` | array | BPE-Merge-Regeln |
| `tokenizer.ggml.vocab_size` | uint32 | Vokabulargröße (Alternative zu `{arch}.vocab_size`) |
| `tokenizer.chat_template` | string | Jinja2-Chat-Template für Prompt-Formatierung |

---

## Capability-Erkennung (MCP/Tool-Calls, Thinking)

Der Scanner leitet zusätzliche Fähigkeiten heuristisch aus vorhandenen Metadaten ab (`detection.py`), da GGUF dafür keine dedizierten Felder vorsieht:

### Tool-Call- / MCP-Unterstützung (`detect_mcp`)

Erkannt über:
- `tokenizer.chat_template` enthält `tool_call`, `function_call`, `<|tool|>` oder `[tool_calls]`
- **oder** `general.tags` enthält `tool`, `function-call` oder `mcp`

### Thinking-/Reasoning-Modus (`detect_thinking`)

Erkannt über (in dieser Reihenfolge):
1. `tokenizer.chat_template` enthält `<think>`
2. `general.tags` enthält `think` oder `reason`
3. Fallback: Name oder Dateiname matcht ein Reasoning-Muster (`think`, `thinking`, `qwq`, `deepseek-r\d+`, `reason`, `reasoning`, `logic`, `reflect`, `chain`, `cog`)

---

## Gesplittete Modelle (Shards)

Große Modelle können auf mehrere Dateien aufgeteilt werden. Das Namensschema lautet:

```
<Modellname>-<Quantisierung>-<NNNNN>-of-<MMMMM>.gguf
```

Beispiel: `Llama-3.1-70B-Instruct-Q4_K_M-00001-of-00003.gguf`

**Wichtig:**
- Jeder Shard enthält die **vollständigen Metadaten** (repliziert), aber nur einen Teil der Tensoren
- Für VRAM-Berechnungen muss die **Gesamtgröße aller Shards** summiert werden
- Nur der **erste Shard** (`00001`) sollte für Metadaten ausgelesen werden, alle weiteren können übersprungen werden

---

## Multimodale Projektoren (mmproj)

Multimodale Modelle (z.B. LLaVA, BakLLaVA) haben einen separaten Vision-Projektor als eigene GGUF-Datei. Erkennungsmerkmale:

- Dateiname beginnt mit `mmproj-`
- **oder** `general.type == "projector"`

Spezifische Metadatenfelder:

| Feld | Beschreibung |
|------|--------------|
| `clip.vision.image_size` | Eingabebildgröße (z.B. 336) |
| `clip.vision.patch_size` | Patch-Größe für ViT (z.B. 14) |
| `clip.vision.embedding_length` | Embedding-Dimension des Vision-Encoders |
| `clip.vision.feed_forward_length` | FFN-Größe |
| `clip.vision.block_count` | Anzahl der Vision-Transformer-Blöcke |
| `clip.vision.projection_dim` | Ausgabedimension des Projektors |
| `clip.has_llava_projector` | Flag für LLaVA-Projektor-Typ |

---

## Referenzen

- [GGUF Format Specification (ggml-org)](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [llama.cpp GGUF Constants (gguf-py)](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/constants.py)
- [gguf-py Python-Bibliothek (PyPI)](https://pypi.org/project/gguf/)
