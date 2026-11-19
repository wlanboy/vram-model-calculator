# VRAM Model Calculator

Beantwortet die kurze, aber komplexe Frage: **Passt dieses GGUF-Modell in meine GPU?**

Das Tool scannt lokale GGUF-Dateien, liest deren Metadaten aus und berechnet den VRAM-Bedarf für verschiedene Kontextlängen. Dazu kommen Hardware- und Modellfamilien-Übersichten für gängige GPUs und APUs.

https://wlanboy.github.io/vram-model-calculator/

---

## 1. Voraussetzungen

Python 3.14+ und `uv`.

**Option A, als installiertes CLI-Tool:**

```bash
uv build
uv tool install dist/vram_model_calculator-0.1.0-py3-none-any.whl
```

**Option B, direkt aus dem Repo:**

```bash
uv sync
```

Alle Befehle laufen über einen Einstiegspunkt mit Unterbefehlen. Installiert heißt er `vram-model-calculator`, im Repo `uv run main.py`. Beide verhalten sich gleich:

| Unterbefehl | Aufgabe | Alter Kurzname |
|---|---|---|
| `scan` | GGUF-Dateien scannen, `models_cache.json` aktualisieren | `gguf-scanner` |
| `check` | GGUF-Dateien auf Integrität prüfen | `gguf-checker` |
| `calculate` | VRAM-Bedarf im Terminal ausgeben | `vram-calculator` |
| `hf-copy` | GGUF-Dateien aus dem HF-Cache ins Modellverzeichnis kopieren | `hf-copy` |
| `hardware` | Tabellen der Hardware-Seiten erzeugen | |
| `families` | Navigation zu den Modellfamilien in `index.html` erzeugen | |

Die alten Kurznamen bleiben als Aliase erhalten, `gguf-scanner --prune` entspricht also `vram-model-calculator scan --prune`.

## 2. GGUF-Modelle scannen

Ohne Argument scannt der Scanner die Standard-Verzeichnisse:

- `~/LMStudio/models/`
- `~/.lmstudio/models/`
- `~/.cache/huggingface/hub/`
- `~/models`
- `/models`
- `/wdblack/models`

Fehlende Pfade wie ein nicht eingehängtes `/wdblack` überspringt er, ohne den Scan abzubrechen. Die Liste lässt sich in der Konfigurationsdatei ändern, siehe Abschnitt 5.

```bash
vram-model-calculator scan                  # installiert
uv run main.py scan                         # aus dem Repo
uv run main.py scan ~/LMStudio/models/      # explizites Verzeichnis statt der Standard-Verzeichnisse
uv run main.py scan /pfad/zu/modell.gguf    # einzelne Datei
```

Cache-Einträge, deren Datei der Scan nicht gefunden hat, bleiben standardmäßig erhalten, denn ein nicht eingehängtes Laufwerk sieht genauso aus wie gelöschte Modelle. `--prune` entfernt sie. Die Option nur verwenden, wenn alle Modell-Verzeichnisse erreichbar sind:

```bash
uv run main.py scan --prune
```

Der Scanner liest die Metadaten jeder GGUF-Datei (Architektur, Layer-Anzahl, Embedding-Dimension, Quantisierung usw.) und speichert sie in `models_cache.json`. Bereits gescannte Dateien überspringt er beim nächsten Aufruf. Er liest nur neue, geänderte oder unvollständig ausgewertete Dateien neu ein.

```
🔍 5 Modelle werden analysiert...
GGUF Scan: 100%|████████████| 5/5 [00:03<00:00]
💾 Cache v12 gespeichert unter 'models_cache.json' (28 Einträge).
```

Nach einem Update der Erkennungslogik erzwingen zwei Optionen einen neuen Scan:

```bash
uv run main.py scan --rescan-all             # alle gefundenen Modelle neu einlesen
uv run main.py scan --rescan qwen            # nur Cache-Keys, die "qwen" enthalten
uv run main.py scan --rescan '*/gemma-4-*'   # Glob-Muster, mehrfach angebbar
```

Pfade mit `heretic`, `heresy`, `uncensored`, `abliterated`, `huihui` oder `dolphin` überspringt der Scanner. `--no-skip` schaltet diesen Filter für einen Lauf ab, `skip_patterns` in der Konfigurationsdatei ersetzt die Liste dauerhaft.

## 3. VRAM-Bedarf im Terminal anzeigen

```bash
vram-model-calculator calculate
uv run main.py calculate
```

Ohne Optionen gibt der Rechner für jedes Modell eine Matrix aus. Die Zeilen sind Kontextlängen von 8k bis 1M Token in Zweierpotenzen, die Spalten GPUs mit 6, 12, 16 und 24 GB:

```
🤖 Kubernetes Istio Model
Arch: qwen3 | Quant: Q4_K_M | Size: 2.33 GB | Compute: 0.30 GB | Overhead: 0.50 GB
Kontext  | KV-Cache   | 6 GB       | 12 GB      | 16 GB      | 24 GB
8k       |     0.70GB | 🟢   3.8G | 🟢   3.8G | 🟢   3.8G | 🟢   3.8G
...
512k     |    45.00GB | 🔴  48.6G | 🔴  48.6G | 🔴  48.6G | 🔴  48.6G
```

Diese Optionen steuern Berechnung und Ausgabe:

| Option | Wirkung |
|---|---|
| `--gpu 12 --gpu 24` | GPU-Größen in GB, mehrfach angebbar |
| `--ctx 8k,32k,128k` | Kontextlängen pro Session, `k` = 1024, `M` = 1024² |
| `--sessions N` | Parallele Sessions, entspricht llama.cpp `-np N` |
| `--cache-type q8_0` | KV-Cache-Typ `f16` (Default), `q8_0` oder `q4_0` |
| `--overhead 0.3` | Fester Overhead pro GPU in GB, Default 0,5 |
| `--filter qwen` | Nur Modelle, deren Name, Architektur oder Cache-Key den Text enthält |
| `--fits 24` | Nur Modelle, die beim kleinsten gewählten Kontext in 24 GB passen, knapp passende zählen mit |
| `--compact` | Eine Zeile pro Modell, eine Tabelle pro Kontextlänge |
| `--json` | Ausgabe als JSON für Skripte |
| `--cache PATH` | Anderer Pfad zu `models_cache.json` |

Beispiel: Welche Modelle laufen mit 32k Kontext und zwei Sessions in 16 GB?

```bash
uv run main.py calculate --ctx 32k --sessions 2 --fits 16 --compact --gpu 16
```

## 4. Interaktive Webansicht öffnen

```bash
uv run python -m http.server 8000
```

Dann im Browser [http://localhost:8000](http://localhost:8000) öffnen.

Die Seite lädt `models_cache.json`, berechnet alle VRAM-Werte im Browser und zeigt eine filterbare, sortierbare Tabelle. `fetch()` braucht einen HTTP-Server, eine per `file://` geöffnete `index.html` lädt keine Daten.

## 5. Konfiguration

Die Datei `~/.config/vram-model-calculator/config.toml` ist optional. `$XDG_CONFIG_HOME` verschiebt sie, `VRAM_CALC_CONFIG` zeigt auf eine beliebige andere Datei.

```toml
scan_dirs = ["~/models", "/mnt/nas/models"]   # Default-Verzeichnisse für scan und check
copy_dest = "~/models"                         # Ziel von hf-copy
skip_patterns = []                             # Namensfilter des Scanners, [] = keiner
cache_file = "~/models/models_cache.json"      # Pfad zum Cache
```

Fehlende Schlüssel behalten ihren Default. Ist die Datei ungültig, bricht der Befehl mit einer Meldung ab.

Den Pfad zum Cache bestimmt die erste dieser Angaben, die gesetzt ist:

1. Option `--cache PATH` bei `scan` und `calculate`
2. Umgebungsvariable `VRAM_CALC_CACHE`
3. `cache_file` aus der Konfigurationsdatei
4. `models_cache.json` im aktuellen Verzeichnis

Ohne weitere Angaben arbeitet das Tool also im Repo-Verzeichnis, wie die GitHub-Pages-Seite es erwartet. Ein installiertes Tool findet den Cache über `VRAM_CALC_CACHE` oder `cache_file` aus jedem Verzeichnis. `model-metadata.txt` mit den Rohdaten unvollständig gelesener Modelle legt der Scanner neben die Cache-Datei.

---

## Werkzeuge im Detail

### `scan`: Metadaten-Scanner

Scannt Verzeichnisse rekursiv nach `.gguf`-Dateien, unabhängig von Groß- und Kleinschreibung der Endung, und extrahiert deren GGUF-Metadaten über den Parser in [`_model.py`](vram_model_calculator/_model.py). Die Dateisuche und die Shard-Gruppierung teilt er sich mit `check` ([`files.py`](vram_model_calculator/files.py)).

**Was der Scanner liest:**

| Feld | Beschreibung |
|---|---|
| `arch` | Modellarchitektur (`llama`, `qwen3`, `mistral`, …) |
| `n_layers` | Anzahl der Transformer-Blöcke |
| `n_embd` | Embedding-Dimension (Hidden Size) |
| `n_heads` / `n_kv_heads` | Attention-Heads und KV-Heads, bei pro Layer variierenden Werten das Maximum, `null` bei reinen SSMs |
| `head_dim_k` / `head_dim_v` (`_swa`) | Head-Dimension für Keys und Values aus `attention.key_length`/`value_length`, Fallback `n_embd / n_heads`. Die `_swa`-Varianten gelten für die Sliding-Window-Layer (Gemma 4) |
| `kv_layers` | Anzahl der Layer mit eigenem KV-Cache, ohne Recurrent-Layer von Hybriden und Layer mit geteiltem KV-Cache |
| `kv_bytes_per_ctx_token` | Vorberechnete KV-Cache-Größe pro Kontext-Token in Bytes für die vollen Attention-Layer (siehe `ModelShape` in `_model.py`). Einzige Quelle der KV-Cache-Formel, `vram_calculator.py` und `filter.js` multiplizieren den Wert nur mit der Kontextlänge |
| `kv_bytes_per_ctx_token_swa` / `swa_window` / `swa_layers` | Bei Sliding-Window-Attention (Gemma 3/4, Cohere2, gpt-oss, …): Bytes pro Token für die lokalen Layer, gedeckelt auf `min(ctx, swa_window)` |
| `recurrent_layers` / `recurrent_state_bytes` | Bei SSM- und Hybrid-Modellen: Anzahl der Recurrent-Layer und Größe des Recurrent-States pro Session in Bytes (siehe `RecurrentShape` in `_model.py`), sonst `0` |
| `n_ff` / `n_ff_exp` | Breite des Feed-Forward-Blocks, bei MoE-Modellen die Breite pro Experte |
| `n_experts` / `n_experts_used` | MoE-Parameter, falls vorhanden |
| `vocab_size` | Größe des Vokabulars |
| `compute_buffer_bytes` | Geschätzter Compute-Buffer von llama.cpp ohne kontextabhängigen Anteil, siehe VRAM-Formel |
| `quant` | Quantisierungstyp (`Q4_K_M`, `Q8_0`, `F16`, …), abgeleitet aus `gguf.constants.LlamaFileType` |
| `n_ctx_orig` | Trainings-Kontextfenster des Modells |
| `file_size_gb` | Dateigröße in GB, entspricht dem VRAM der Gewichte. Bei Shards die Summe aller Teile |
| `mcp` / `thinking` | Heuristisch erkannte Tool-Calling- bzw. Reasoning-Fähigkeit |
| `has_missing_fields` | `true`, wenn Pflichtfelder fehlten. Der nächste Scan liest die Datei neu ein |

**Inkrementeller, versionierter Cache.** Der Scanner liest eine Datei nur neu ein, wenn sie noch nicht im Cache steht, sich ihre Größe geändert hat, sie beim letzten Mal `has_missing_fields: true` bekam oder `--rescan`/`--rescan-all` sie erfasst. Der Cache trägt eine `_version`, die jeder Schreibvorgang hochzählt. Beim Laden bereinigt der Scanner Modellnamen, wandelt Zahlen aus sehr alten Cache-Versionen, die als String gespeichert waren, in `int` um und ergänzt fehlende Felder wie `kv_bytes_per_ctx_token` und `compute_buffer_bytes` aus den gespeicherten Dimensionen. Dafür braucht er keinen Zugriff auf die GGUF-Dateien.

**Besonderheiten:**
- Mehrteilige Modelle (`-00001-of-00005.gguf`) behandelt der Scanner als eine Einheit. Er parst nur den ersten Shard und summiert die Größe über alle Shards
- Dateien mit `mmproj` im Namen erkennt er als Vision-Projektor und speichert sie separat (`"type": "mmproj"`)
- Den KV-Cache berechnet er wie llama.cpp pro Layer: KV-Heads pro Layer (`attention.head_count_kv` als Array, z. B. Gemma 4), eigene Head-Dimensionen für SWA-Layer (`key_length_swa`), Layer ohne eigenen Cache (`attention.shared_kv_layers`, Gemma 3n/4)
- Fehlt `sliding_window_pattern` in der Datei (Gemma 3, gpt-oss), gilt der Architektur-Default aus llama.cpp (`SWA_DEFAULT_PATTERN` in `_model.py`). Unbekannte Architekturen ohne Pattern bekommen wie in llama.cpp kein SWA
- Bei Hybriden aus `SSM_ARCHS` (Qwen3.5/Qwen3-Next, LFM2, Nemotron-H, Granite-Hybrid, Jamba u. a.) zählen nur die Attention-Layer zum KV-Cache. Der Scanner erkennt sie an `0`-Einträgen im `head_count_kv`-Array, an `attention.recurrent_layers` oder an `full_attention_interval`. Reine SSMs (Mamba, RWKV) haben `n_kv_heads = null` und damit keinen KV-Cache. CLI und Webseite werten `n_kv_heads` `null` oder `0` gleich als reines SSM (`is_pure_ssm` in `_model.py`)
- SSM-Layer belegen pro Session einen festen Recurrent-State, der nicht mit dem Kontext wächst. Der Scanner berechnet ihn wie llama.cpp (`n_embd_r` + `n_embd_s` in `llama-hparams.cpp`, f32) aus `ssm.*`, `wkv.head_size`, `shortconv.l_cache` und `kda.head_dim`. Einträge von SSM-Architekturen ohne `recurrent_state_bytes` liest der Scanner beim nächsten Lauf einmal neu ein
- Fehlende Metadaten wie `vocab_size` ermittelt er über Fallback-Methoden. Modelle mit weiterhin fehlenden Feldern protokolliert er in `model-metadata.txt`

---

### `check`: Integritätsprüfung

Prüft `.gguf`-Dateien auf strukturelle Korrektheit und Vollständigkeit, unabhängig vom Metadaten-Cache. Das hilft nach einem Download oder Kopiervorgang, um abgebrochene oder beschädigte Dateien zu finden, bevor llama.cpp sie lädt.

```bash
uv run main.py check                                   # scan_dirs aus der Konfiguration
uv run main.py check /models
uv run main.py check ~/LMStudio/models/modell.gguf
```

**Was geprüft wird:**

| Prüfung | Erkennt |
|---|---|
| Magic-Bytes (`GGUF`) und Version | Falsche oder fremde Dateien, komplett beschädigte Header |
| Header- und Tensor-Info-Parsing | Beschädigte Metadaten, z. B. doppelte Tensor-Namen |
| Tensor-Datenbereich gegen tatsächliche Dateigröße | Abgebrochene Downloads, abgeschnittene Dateien |
| Shard-Vollständigkeit (`-00001-of-00005.gguf`) | Fehlende Teile bei mehrteiligen Modellen |

**Status je Datei:** `ok` (vollständig und valide), `incomplete` (Datei bricht mitten in den Tensor-Daten ab oder ein Shard fehlt) oder `corrupt` (ungültiger Header oder Metadaten). Sobald eine Datei nicht `ok` ist, endet der Prozess mit Exit-Code `1`. Damit eignet er sich auch für Skripte und CI.

---

### `hf-copy`: GGUF-Dateien aus dem HF-Cache kopieren

Listet alle `.gguf`-Dateien im Huggingface-Hub-Cache (`~/.cache/huggingface/hub/`) nummeriert auf und kopiert eine Auswahl nach `<ziel>/<org>/<repo>/`. Das Ziel ist `copy_dest` aus der Konfiguration, Default `/wdblack/models`.

```bash
uv run main.py hf-copy                    # listet auf, fragt interaktiv nach der Auswahl
uv run main.py hf-copy 1,3,5-7            # kopiert die Einträge 1, 3, 5, 6, 7
uv run main.py hf-copy all                # kopiert alle gefundenen Dateien
uv run main.py hf-copy --dry-run all      # zeigt nur an, kopiert nichts
uv run main.py hf-copy --dest /pfad all   # abweichendes Zielverzeichnis
```

Dateien, die am Ziel mit gleichem Namen und gleicher Größe schon liegen, überspringt das Tool.

---

### `calculate`: Terminal-Rechner

Liest `models_cache.json` und gibt den VRAM-Bedarf jedes LLM-Modells aus. Die Optionen stehen in Abschnitt 3.

**VRAM-Formel:**

```
Gewichte        = file_size_gb
KV-Cache        = (kv_bytes_per_ctx_token × ctx
                   + kv_bytes_per_ctx_token_swa × min(ctx, swa_window))
                  × kv_faktor / 1024³
State           = recurrent_state_bytes / 1024³
Compute-Buffer  = (compute_buffer_bytes + ctx × Sessions × 1024) / 1024³
Overhead        = 0,5 GB (einstellbar)
Gesamt          = Gewichte + (KV-Cache + State) × Sessions + (Compute-Buffer + Overhead) × GPUs
```

`ctx` ist der Kontext pro Session. In llama.cpp teilt `-np N` den Kontext `-c` auf N Slots auf: Bei `-c 32768 -np 4` hat jeder Slot 8k. Die Einstellung „8k Kontext, 4 Sessions“ in CLI und Webseite entspricht also `-c 32768 -np 4`. Die Webseite zeigt den passenden Aufruf neben der Sessions-Auswahl an, die CLI in der Kopfzeile.

Die Spalte „KV-Cache“ in CLI und Webseite enthält den Recurrent-State mit, die CLI nennt ihn zusätzlich in der Kopfzeile des Modells. llama.cpp hält den State immer in f32, `kv_faktor` gilt für ihn nicht.

Die Bytes pro Token in `models_cache.json` gelten für einen KV-Cache in f16. `kv_faktor` rechnet sie auf den gewählten KV-Cache-Typ um (`KV_CACHE_TYPES` in `vram_calculator.py` und `filter.js`):

| KV-Cache-Typ | Bytes pro 32 Elemente | `kv_faktor` |
|---|---|---|
| `f16` | 64 | 1 |
| `q8_0` | 34 | 34/64 ≈ 0,53 |
| `q4_0` | 18 | 18/64 ≈ 0,28 |

Die Auswahl entspricht in llama.cpp den Optionen `--cache-type-k` und `--cache-type-v` mit demselben Typ. Einen quantisierten V-Cache nutzt llama.cpp nur mit Flash Attention.

Der SWA-Summand entfällt (`swa_window` ist `null`), solange das Modell keine Sliding-Window-Attention nutzt. `_model.py` berechnet `kv_bytes_per_ctx_token` und `kv_bytes_per_ctx_token_swa` beim Scan (`ModelShape.kv_bytes_per_ctx_token()`/`kv_bytes_per_ctx_token_swa()`) und speichert sie im Cache:

```
Globale Layer:        Σ über alle globalen Layer mit eigenem KV-Cache: n_kv_heads(layer) × (head_dim_k + head_dim_v) × 2
Sliding-Window-Layer: Σ über alle SWA-Layer: n_kv_heads(layer) × (head_dim_k_swa + head_dim_v_swa) × 2, gedeckelt auf swa_window Tokens
MLA (DeepSeek2 u.a.): n_layers × mla_kv_dim × 2
Recurrent-/geteilte Layer, reine SSMs: 0
```

**Compute-Buffer.** llama.cpp reserviert neben Gewichten und KV-Cache einen Puffer für die Zwischenergebnisse eines Micro-Batches und meldet ihn im Log als `compute buffer size`. Der Scanner schätzt ihn in `compute_buffer_bytes()` (`_model.py`) für `n_ubatch = 512` und f32-Aktivierungen:

```
compute_buffer_bytes = 512 × 4 × max(vocab_size + n_embd, 2 × ffn_breite + 2 × n_embd)
ffn_breite           = n_ff, bei MoE n_ff_exp × n_experts_used
```

Der erste Term sind die Logits am Ende des Graphen, der zweite der Feed-Forward-Block. ggml verwendet Speicher wieder, darum zählt nur der größere. Bei großen Vokabularen wie Llama 3 (128k Token) dominieren die Logits mit rund 260 MB. Mit dem Kontext wächst bei Flash Attention nur die KQ-Maske, eine f16-Zahl pro KV-Zelle und Ubatch-Token, also 1024 Bytes pro Token im Gesamtkontext. Ohne Flash Attention fällt der Puffer größer aus.

**Overhead.** Der Treiber-Kontext von CUDA, SYCL oder Vulkan und kleine llama.cpp-Puffer belegen pro GPU zusätzlich etwa 0,3 bis 0,5 GB. Der Default ist 0,5 GB, `--overhead` und das Feld „Overhead“ der Webseite ändern ihn. Bei mehreren GPUs liegen Compute-Buffer und Overhead auf jeder Karte.

**Farbcodierung:**

| Symbol | Bedeutung |
|---|---|
| 🟢 | Passt bequem, höchstens 85 % der GPU-Kapazität |
| 🟡 | Passt knapp, höchstens 100 % |
| 🔴 | Passt nicht |

---

### `index.html`: Interaktive Webansicht

Eine Browser-Anwendung ohne Build-Schritt aus vier Dateien:

| Datei | Aufgabe |
|---|---|
| `index.html` | Struktur: Tabelle, Bedienelemente, Navigation zu Hardware und Modellfamilien |
| `filter.js` | JSON laden, VRAM berechnen, filtern, sortieren, Zustand in der URL halten |
| `common.js` | HTML-Escaping und sortierbare Spaltenköpfe, gemeinsam mit den Hardware-Seiten |
| `style.css` | Dark-Mode-Design |

**Funktionen:**

- **Kontext.** Frei wählbare Kontextlänge pro Session in 1024er-Schritten, Standard 8192.
- **Sessions.** Multipliziert KV-Cache und Recurrent-State. Darunter steht der passende llama.cpp-Aufruf, z. B. „entspricht `-c 65536 -np 2`“. Die GPU-Zellen zeigen, wie viele Sessions neben Gewichten, Compute-Buffer und Overhead Platz hätten.
- **KV-Cache.** Typ `f16`, `q8_0` oder `q4_0`, skaliert KV-Cache, Gesamtbedarf und die Zahl paralleler Sessions.
- **GPUs.** Eine bis vier gleiche Karten. Die Spalten zeigen dann z. B. „2× 24 GB“, Compute-Buffer und Overhead zählen pro Karte.
- **Overhead.** Fester Aufschlag pro GPU in GB, Default 0,5.
- **Quants gruppieren.** Fasst Quantisierungen desselben Modells (gleicher Name, gleiche Größenangabe) zu einer Zeile zusammen. Die Zeile zeigt die größte Quantisierung, die in die gewählte GPU passt, ein Badge nennt die übrigen.
- **Feature-Filter.** MCP (Tool-Calling) und Thinking/Reasoning, heuristisch aus den Modelldaten erkannt.
- **Suche.** Freitext über Modellname und Architektur.
- **Sichtbare VRAM-Spalten.** Zehn GPU-Größen von 4 bis 80 GB als Pills, standardmäßig 6 und 12 GB. Über „+ eigene Größe“ kommen weitere Größen wie 20 oder 96 GB dazu, etwa für Unified Memory mit nutzbarem Anteil.
- **Rote Modelle ausblenden.** Blendet Modelle aus, die in eine der sichtbaren GPU-Spalten nicht passen.
- **Aufschlüsselung.** Der Pfeil vor dem Modellnamen klappt eine Zeile mit Gewichten, KV-Cache, Recurrent-State, Compute-Buffer, Overhead und gegebenenfalls `mmproj` auf, dazu den passenden `llama-server`-Aufruf.
- **Sortierung.** Jede Spalte ist per Klick, Enter oder Leertaste auf- und absteigend sortierbar.

Alle Einstellungen stehen im URL-Hash, z. B. `#ctx=32768&s=2&kv=q8_0&gpu=16,24`. Neuladen behält sie, und ein Link gibt die Ansicht weiter. Ein Badge „ctx“ am Modellnamen warnt, wenn der gewählte Kontext das Trainings-Kontextfenster überschreitet.

`filter.js` lädt `models_cache.json` ohne Browser-Cache (`cache: "no-cache"`), damit nach einem Push der neue Stand erscheint. Die Konstanten `TIGHT_FIT_RATIO`, `KV_CACHE_TYPES`, `DEFAULT_OVERHEAD_GB` und `COMPUTE_MASK_BYTES_PER_CTX_TOKEN` stehen in Python und JavaScript, `tests/test_vram_calculator.py` prüft, dass sie übereinstimmen.

---

### `hardware/`: Hardware-Kompatibilitätsübersichten

Kompatibilitätstabellen für konkrete GPUs und APUs: welche Modelle darauf komfortabel laufen, welche knapp, welche gar nicht. `hardware/index.html` listet alle Geräte als Karten mit Kurzstatistik, jede Unterseite (`hardware/<Gerät>/index.html`) zeigt die volle Tabelle.

| Gerät | Typ |
|---|---|
| GMKtec EVO-X2 | Ryzen AI Max+ 395, 128 GB Unified Memory |
| Intel Arc Pro B60 | Workstation-GPU, 24 GB VRAM |
| Intel Arc B580 | Consumer-GPU, 12 GB VRAM |

Die Daten jedes Geräts stehen in `hardware/<Gerät>/device.json`:

```json
{
  "title": "Intel Arc B580 (12 GB)",
  "vram_gb": 12,
  "ctx": 8192,
  "models": [
    {"name": "Qwen3 4B", "params_b": 4.0, "type": "Chat", "quant": "Q4_K_M",
     "vram_gb": 2.7, "speed_ts": 37.3, "ctx_label": "41K", "ctx_num": 41,
     "status": "well", "status_text": "RUNS WELL", "grade": "A", "score": 78}
  ]
}
```

`vram_gb` ist der nutzbare Speicher des Geräts, `ctx` der Kontext für die VRAM-Spalte. Optional sind `cache_type`, `overhead_gb` sowie pro Modell `cache_key` und `source`. Geschwindigkeit und Bewertung sind Messwerte und werden in dieser Datei von Hand gepflegt.

```bash
uv run main.py hardware           # MODELS-Arrays und Übersichtskarten neu schreiben
uv run main.py hardware --check   # nur prüfen, Exit-Code 1 bei veralteten Seiten
uv run main.py hardware --init    # device.json aus einer handgeschriebenen Seite anlegen
```

Der Generator ordnet jedes Modell einem Cache-Eintrag zu, entweder über `cache_key` oder über gleichen Namen und gleiche Quantisierung, wobei er Groß- und Kleinschreibung sowie Sonderzeichen ignoriert. Für zugeordnete Modelle berechnet er VRAM und Kontext mit derselben Formel wie `calculate`. Modelle ohne Treffer im Cache behalten den VRAM-Wert aus `device.json`. Die Übersichtskarten zählen Modelle mit Status `great` als „exzellent“ und `well` als „gut“.

---

### `modell-familien/`: Modellfamilien-Steckbriefe

Übersichtsseiten je Modellfamilie mit Architektur, typischen Größen, Besonderheiten und Empfehlungen. `uv run main.py families` schreibt die Links in der Navigation von `index.html` zwischen die Marker `<!-- BEGIN FAMILIES -->` und `<!-- END FAMILIES -->`, alphabetisch sortiert. Der Linktext kommt aus `<meta name="nav-label" content="…">` der Seite, sonst aus dem `<title>` bis zum ersten Gedankenstrich. Eine neue Seite braucht also nur die Datei in `modell-familien/` und einen Lauf von `families`. Ein Test schlägt fehl, solange die Navigation veraltet ist.

---

### `gguf-format.md`: Referenz zum GGUF-Dateiformat

Technische Übersicht über das GGUF-Format: Aufbau von Header, Metadaten und Tensordaten, `general.type`-Werte, architekturspezifische Metadatenfelder. Das Dokument erklärt, welche Felder der Scanner ausliest und warum. Die Tools lesen es nicht.

---

## Installation als CLI-Tool (Wheel)

### Wheel bauen

```bash
uv build
# Erstellt:
#   dist/vram_model_calculator-0.1.0-py3-none-any.whl
#   dist/vram_model_calculator-0.1.0.tar.gz
```

### Global installieren

```bash
uv tool install dist/vram_model_calculator-0.1.0-py3-none-any.whl
```

Danach stehen `vram-model-calculator` und die Aliase `gguf-scanner`, `gguf-checker`, `vram-calculator` und `hf-copy` systemweit bereit. Damit das installierte Tool außerhalb des Repos denselben Cache nutzt, `VRAM_CALC_CACHE` setzen oder `cache_file` in der Konfigurationsdatei eintragen:

```bash
export VRAM_CALC_CACHE=~/git/vram-model-calculator/models_cache.json
vram-model-calculator scan
vram-model-calculator calculate --fits 24 --compact
```

### Ohne Installation testen

```bash
uv run --with dist/vram_model_calculator-0.1.0-py3-none-any.whl vram-model-calculator scan ~/LMStudio/models/
uv run --with dist/vram_model_calculator-0.1.0-py3-none-any.whl vram-model-calculator calculate
```

### Neu installieren oder aktualisieren

```bash
uv build
uv cache clean vram-model-calculator   # sonst zieht uv die alte Version
uv tool install --force dist/vram_model_calculator-0.1.0-py3-none-any.whl
```

### Deinstallieren

```bash
uv tool uninstall vram-model-calculator
```

---

## CI

`.github/workflows/ci.yml` läuft bei jedem Push und Pull Request auf `main`:

1. Abhängigkeiten mit `uv sync --frozen` exakt nach `uv.lock` installieren
2. Lint und Syntax-Check mit `ruff check .`
3. Typprüfung mit `pyright .`
4. Tests mit `pytest`
