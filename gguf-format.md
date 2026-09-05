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

**Nicht vom Scanner ausgewertet, aber für exakte KV-Cache-VRAM-Schätzung relevant:** `{arch}.attention.sliding_window`/`.sliding_window_pattern` (bei SWA-Hybriden wie Gemma3 ist der effektive KV-Cache pro Layer kleiner als `context_length` vermuten lässt), `.key_length_mla`/`.value_length_mla`/`.kv_lora_rank`/`.kv_lora_rank_swa` (Multi-Head-Latent-Attention, z.B. DeepSeek2/Kimi/Bailing), `.key_length_swa`/`.value_length_swa`, `.shared_kv_layers` (Layer mit geteiltem KV-Cache). Modelle mit diesen Feldern werden von `_model.py` aktuell mit einem einzigen globalen `n_kv_heads`/`key_length` gerechnet, was den tatsächlichen KV-Cache-Bedarf über- oder unterschätzen kann.

### RoPE-Parameter

| Feld | Typ | Beschreibung |
|------|-----|--------------|
| `{arch}.rope.dimension_count` | uint32 | Anzahl der RoPE-Dimensionen pro Head |
| `{arch}.rope.freq_base` | float32 | Basisfrequenz (Standard ~10.000; erweiterte Kontextmodelle nutzen bis 1.000.000) |
| `{arch}.rope.scaling.factor` | float32 | Linearer Skalierungsfaktor für Kontexterweiterung (das früher hier dokumentierte `rope.scale_linear` existiert in der aktuellen Spec nicht mehr) |
| `{arch}.rope.dimension_count_swa` / `.freq_base_swa` | uint32/float32 | Separate RoPE-Parameter für Sliding-Window-Attention-Layer |
| `{arch}.rope.dimension_sections` | array | Sektionierte RoPE-Dimensionen (z.B. Multi-axis RoPE/MRoPE) |
| `{arch}.rope.scaling.attn_factor` / `.original_context_length` / `.finetuned` | — | Weitere Skalierungsmetadaten |
| `{arch}.rope.scaling.yarn_ext_factor` / `.yarn_attn_factor` / `.yarn_beta_fast` / `.yarn_beta_slow` | float32 | YaRN-spezifische Kontexterweiterungs-Parameter |

### MoE-Parameter (Mixture of Experts)

| Feld | Typ | Beschreibung |
|------|-----|--------------|
| `{arch}.expert_count` | uint32 | Gesamtzahl der Experten |
| `{arch}.expert_used_count` | uint32 | Aktivierte Experten pro Token-Durchlauf |
| `{arch}.expert_shared_count` | uint32 | Anzahl fest aktiver "shared experts" zusätzlich zu den gerouteten |
| `{arch}.expert_group_count` | uint32 | Experten-Gruppierung (z.B. für gruppiertes Top-k-Routing) |
| `{arch}.moe_every_n_layers` | uint32 | MoE nur in jeder n-ten Schicht aktiv |
| `{arch}.moe_latent_size` | uint32 | Latent-Dimension für MoE-Kompression |

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
| `falcon-h1` | Hybrid | Falcon H1 (Hybrid Head) |
| `granitehybrid` | Hybrid | IBM Granite Hybrid |
| `plamo2` | Hybrid | PLaMo 2 |
| `plamo3` | Hybrid | PLaMo 3 |
| `qwen3next` | Hybrid | Qwen3-Next |
| `lfm2` | Hybrid | LiquidAI LFM2 |
| `lfm2moe` | Hybrid | LiquidAI LFM2 (MoE) |
| `nemotron_h` | Hybrid | Nvidia Nemotron-H |
| `nemotron_h_moe` | Hybrid | Nvidia Nemotron-H (MoE) |
| `qwen35` | Hybrid | Qwen3.5 (lädt SSM-Felder wie `ssm_d_state`/`ssm_dt_rank`) |
| `qwen35moe` | Hybrid | Qwen3.5 (MoE-Variante) |
| `qwen4exp` | Hybrid | Qwen4-Experimental (Lightning-Attention-artiges Layer-Muster) |
| `kimi-linear` | Hybrid | Kimi Linear (eigener KDA-Namensraum, RoPE-los) |
| `kimi-k3` | Hybrid | Kimi K3 (nutzt dieselbe KDA-Mechanik wie kimi-linear) |
| `bailingmoe3` | Hybrid | BailingMoE v3 (KDA-Namensraum) |
| `minimax-01` | Hybrid | MiniMax-Text-01 (rekurrente Linear-Attention-Layer pro Layer-Flag) |

Diese Liste (`SSM_ARCHS` in `_model.py`) wächst mit jeder neuen State-Space-/Hybrid-Architektur, die llama.cpp unterstützt, und muss bei Bedarf ergänzt werden. Zuletzt gegen llama.cpp-Commit `6a1a922d` (2026-09) abgeglichen — siehe `gguf-format-research-2026-09.md` für Details und Belege.

**Korrigierter Namens-Bug (2026-09):** Die tatsächlichen `general.architecture`-Strings laut `llama-arch.cpp` sind `falcon-h1` (mit Bindestrich) und `granitehybrid` (ohne Trenner) — die zuvor im Projekt verwendeten Schreibweisen `falcon_h1`/`granite_hybrid` matchten nie gegen echte GGUF-Dateien dieser Architekturen, wodurch fälschlich eine „fehlende Felder"-Warnung ausgelöst wurde.

**Explizit geprüft und NICHT hybrid:** `minimax-m2`, `minimax-m3`, `granite_swa`, `afmoe`, `gpt-oss` laden keine SSM-/rekurrenten Felder und werden korrekt als normale Attention-Architekturen behandelt. `nanbeige` ist unklar (kopiert rekurrenz-artige Layer-Arrays für einen Weight-Sharing-/Loop-Mechanismus, lädt aber keine SSM-Keys) und wurde bewusst nicht aufgenommen.

**Sonderfall Text-Diffusions-LLMs:** `dream`, `llada`, `llada-moe`, `rnd1` sind nicht-autoregressive Text-Diffusionsmodelle, aber architektonisch normale Transformer mit Standard-`block_count`/Attention-Feldern (keine SSM-Felder, kein globaler `attention.head_count_kv`-Sonderfall). Sie gehören weder zu `SSM_ARCHS` noch zu `DIFFUSION_ARCHS` und sollten vom Scanner wie gewöhnliche LLM-Architekturen erfasst werden.

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

Der Wert von `general.file_type` ist ein Integer, der über die Enum-Klasse `LlamaFileType` aus dem `gguf`-Paket aufgelöst werden kann (vollständige Liste, Stand llama.cpp-Commit `6a1a922d`, 2026-09):

| ID | Name | Beschreibung |
|----|------|--------------|
| 0 | F32 | 32-bit Float |
| 1 | F16 | 16-bit Float |
| 2 | Q4_0 | 4-bit Quantisierung (älteres Format) |
| 3 | Q4_1 | 4-bit Quantisierung, Variante 1 |
| 7 | Q8_0 | 8-bit Quantisierung |
| 8 | Q5_0 | 5-bit Quantisierung |
| 9 | Q5_1 | 5-bit Quantisierung, Variante 1 |
| 10 | Q2_K | 2-bit K-Quant |
| 11–13 | Q3_K_S/M/L | 3-bit K-Quant (Small/Medium/Large) |
| 14–15 | Q4_K_S/M | 4-bit K-Quant (Small/Medium) |
| 16–17 | Q5_K_S/M | 5-bit K-Quant (Small/Medium) |
| 18 | Q6_K | 6-bit K-Quant |
| 19–20 | IQ2_XXS/IQ2_XS | I-Quants, sehr niedrige Bitrate |
| 21 | Q2_K_S | 2-bit K-Quant, Small |
| 22–27 | IQ3_XS/IQ3_XXS/IQ1_S/IQ4_NL/IQ3_S/IQ3_M | I-Quants (3-/4-/1-bit-Varianten) |
| 28–29 | IQ2_S/IQ2_M | I-Quants, 2-bit |
| 30–31 | IQ4_XS/IQ1_M | I-Quants (4-/1-bit) |
| 32 | BF16 | Brain Float 16 |
| 33–35 | *(reserviert)* | Ehemals `Q4_0_4_4`/`Q4_0_4_8`/`Q4_0_8_8` (ARM-Repack-Formate); aus GGUF-Dateien entfernt, IDs bleiben unbelegt |
| 36–37 | TQ1_0/TQ2_0 | Ternäre Quantisierung |
| 38 | MXFP4_MOE | Microscaling FP4 für MoE-Layer |
| 39 | NVFP4 | Nvidia FP4-Quantisierung |
| 40–41 | Q1_0/Q2_0 | 1-/2-bit Quantisierung |
| 1024 | GUESSED | Sentinel: `file_type` fehlt in der Datei, vom Loader geschätzt |

Bekannte Diskrepanz (behoben 2026-09): Der interne Fallback in `gguf_fields.py` (greift nur, wenn das `gguf`-PyPI-Paket nicht installiert ist) war ab ID 21 falsch nummeriert und kannte IDs 39–41 nicht — wurde an obige Tabelle angeglichen.

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

Analog zu `mmproj-` gibt es außerdem das Sidecar-Präfix `mtp-` für separate Multi-Token-Prediction-Heads (Speculative-Decoding-Draft-Module) — kein eigener `general.type`-Wert, nur Namenskonvention.

**Wichtig:**
- Jeder Shard enthält die **vollständigen Metadaten** (repliziert), aber nur einen Teil der Tensoren
- Für VRAM-Berechnungen muss die **Gesamtgröße aller Shards** summiert werden
- Nur der **erste Shard** (`00001`) sollte für Metadaten ausgelesen werden, alle weiteren können übersprungen werden

---

## Multimodale Projektoren (mmproj)

Multimodale Modelle (z.B. LLaVA, BakLLaVA) haben einen separaten Vision-Projektor als eigene GGUF-Datei. Erkennungsmerkmale:

- Dateiname beginnt mit `mmproj-`
- **oder** `general.type == "projector"`

Vom Projekt ausgewertete Felder (`get_mmproj_params()` in `_model.py`):

| Feld | Beschreibung |
|------|--------------|
| `clip.vision.image_size` | Eingabebildgröße (z.B. 336) |
| `clip.vision.patch_size` | Patch-Größe für ViT (z.B. 14) |
| `clip.vision.embedding_length` | Embedding-Dimension des Vision-Encoders |
| `clip.vision.feed_forward_length` | FFN-Größe |
| `clip.vision.block_count` | Anzahl der Vision-Transformer-Blöcke |
| `clip.vision.projection_dim` | Ausgabedimension des Projektors |
| `clip.has_llava_projector` | Flag für LLaVA-Projektor-Typ |

### Weitere `clip.vision.*`-Felder (nicht ausgewertet, aber in aktuellen GGUF-Dateien vorhanden)

Neuere Vision-Encoder (Qwen2.5-VL, MiMo-VL, Granite4-Vision u.a.) liefern zusätzlich u.a. `image_min_pixels`/`image_max_pixels`, `preproc_min_tiles`/`preproc_max_tiles`/`preproc_image_size`, `image_mean`/`image_std`, `spatial_merge_size`, `expert_count_per_layer`/`expert_used_count` (MoE-Vision-Encoder), `attention.head_count_kv` (GQA im Vision-Encoder), `attention.head_dim`, `window_size`, `is_deepstack_layers`. Diese werden vom Scanner aktuell nicht gelesen.

### Audio-Encoder: `clip.audio.*` / `clip.gen.audio.*` (neuer Namensraum, nicht unterstützt)

Seit einer neueren llama.cpp-Version existiert analog zu `clip.vision.*` ein eigener Namensraum für Audio-Encoder (Speech-to-Text-Projektoren, z.B. `clip.audio.num_mel_bins`, `.embedding_length`, `.feed_forward_length`, `.block_count`, `.attention.head_count`) sowie `clip.gen.audio.*` für generative Audio-/TTS-Modelle. Ein reiner Audio-`mmproj` (kein `clip.vision.*`) wird von `get_mmproj_params()` aktuell **nicht** erkannt: Die kritischen Felder (`image_size`, `n_embd`, `n_layers`) werden ausschließlich aus `clip.vision.*` gelesen, sodass eine valide Audio-only-Datei fälschlich als „fehlende Felder“ markiert würde. Das Flag `clip.has_audio_encoder` (analog zu `clip.has_llava_projector`) könnte künftig genutzt werden, um Audio-Projektoren zu erkennen und die passenden Felder auszuwerten — bisher nicht implementiert.

---

## Referenzen

- [GGUF Format Specification (ggml-org)](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [llama.cpp GGUF Constants (gguf-py)](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/constants.py)
- [gguf-py Python-Bibliothek (PyPI)](https://pypi.org/project/gguf/)
