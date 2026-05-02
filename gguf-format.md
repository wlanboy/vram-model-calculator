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

### Reine SSM-Architekturen (ohne Attention, kein `n_kv_heads`)

State-Space-Modelle (SSMs) verwenden keine klassischen Attention-Layer. Das Feld `attention.head_count_kv` existiert nicht oder ist nicht relevant:

| Architektur | Beschreibung |
|-------------|--------------|
| `mamba` | Mamba (SSM v1) |
| `mamba2` | Mamba2 (State Space Model v2) |
| `rwkv` | RWKV (ältere Versionen) |
| `rwkv6` | RWKV v6 |
| `rwkv7` | RWKV v7 (Eagle/Finch) |

### Hybride Architekturen (SSM + Attention)

Diese Modelle kombinieren SSM-Blöcke und Attention-Blöcke. Sie benötigen `attention.head_count_kv` für ihre Attention-Schichten und dürfen **nicht** als reine SSMs behandelt werden:

| Architektur | Beschreibung |
|-------------|--------------|
| `jamba` | Jamba (AI21 Labs) |
| `zamba` | Zamba |
| `bamba` | Bamba |
| `falcon_h1` | Falcon H1 (Hybrid Head) |

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
