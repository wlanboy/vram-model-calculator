# VRAM Model Calculator

Beantwortet eine einfache Frage: **Passt dieses GGUF-Modell in meine GPU?**

Das Tool scannt lokale GGUF-Dateien, liest deren Metadaten aus und berechnet den VRAM-Bedarf für verschiedene Kontextlängen — vom kurzen Chat-Einsatz bis hin zu langen Coding-Agent-Läufen. Das Ergebnis lässt sich sowohl im Terminal als auch als interaktive Webseite anzeigen.

---

## Tutorial: von Null zum Ergebnis

### 1. Voraussetzungen

Python 3.11+ und `uv` (empfohlen) oder `pip`:

```bash
# Mit uv (empfohlen)
uv sync

# Oder klassisch
pip install gguf tqdm
```

### 2. GGUF-Modelle scannen

Zeige auf das Verzeichnis, in dem deine `.gguf`-Dateien liegen (LM Studio legt sie z. B. unter `~/LMStudio/models/` ab):

```bash
python gguf-scanner.py ~/LMStudio/models/
```

Der Scanner liest die Metadaten aus jeder GGUF-Datei (Architektur, Layer-Anzahl, Embedding-Dimension, Quantisierung usw.) und speichert alles in `models_cache.json`. Bereits gescannte Dateien werden beim nächsten Aufruf übersprungen — nur neue oder geänderte Dateien werden verarbeitet.

```
🔍 5 Modelle werden analysiert...
GGUF Scan: 100%|████████████| 5/5 [00:03<00:00]
💾 Cache gespeichert unter 'models_cache.json' (28 Einträge).
```

### 3. VRAM-Matrix im Terminal anzeigen (optional)

```bash
python vram-calculator.py
```

Gibt für jedes Modell eine Tabelle aus, die zeigt, wie viel VRAM für jeden Anwendungsfall benötigt wird und ob es in die konfigurierten GPUs passt:

```
🤖 Qwen3-4B-Thinking-2507-Q4_K_M
Arch: qwen3 | Size: 2.33 GB
Usecase      | KV-Cache   | 6GB (Entry) | 12GB (Mid) | 16GB (Pro) | 24GB (Ultra)
Chat (8k)    |     0.19GB | 🟢  2.5G   | 🟢  2.5G  | ...
Agent (512k) |    11.87GB | 🔴 14.2G   | 🟡 14.2G  | ...
```

### 4. Interaktive Webansicht öffnen

```bash
python -m http.server
```

Dann im Browser: [http://localhost:8000](http://localhost:8000)

Die Seite lädt `models_cache.json` direkt, berechnet alle VRAM-Werte im Browser und zeigt eine filterbare Tabelle.

> `fetch()` benötigt einen HTTP-Server — direktes Öffnen der `index.html` per `file://` funktioniert nicht.

---

## Werkzeuge im Detail

### `gguf-scanner.py` — Metadaten-Scanner

Scannt ein Verzeichnis rekursiv nach `.gguf`-Dateien und extrahiert deren GGUF-Metadaten.

**Aufruf:**

```bash
python gguf-scanner.py <pfad>
# Beispiele:
python gguf-scanner.py ~/LMStudio/models/
python gguf-scanner.py .
```

**Was der Scanner liest:**

| Feld | Beschreibung |
|---|---|
| `arch` | Modellarchitektur (`llama`, `qwen3`, `mistral`, …) |
| `n_layers` | Anzahl der Transformer-Blöcke |
| `n_embd` | Embedding-Dimension (Hidden Size) |
| `n_heads` / `n_kv_heads` | Attention-Heads / KV-Heads |
| `n_experts` / `n_experts_used` | MoE-Parameter (falls vorhanden) |
| `quant` | Quantisierungstyp (`Q4_K_M`, `Q8_0`, `F16`, …) |
| `n_ctx_orig` | Trainings-Kontextfenster des Modells |
| `file_size_gb` | Dateigröße in GB (= Gewichts-VRAM) |

**Inkrementeller Cache:**  
`models_cache.json` wird beim nächsten Scan wiederverwendet. Eine Datei wird nur neu gescannt, wenn sie noch nicht im Cache ist oder sich ihre Dateigröße geändert hat. Bei veralteter Cache-Version (`_version`) wird der Cache automatisch neu aufgebaut.

**Besonderheiten:**
- Dateien, die mit `mmproj-` beginnen, werden als Vision-Projektor erkannt und separat gespeichert (`"type": "mmproj"`)
- SSM-Modelle (z. B. LFM2, Nemotron-H) haben `n_kv_heads = 0` — kein KV-Cache
- Fehlende Metadaten (z. B. `vocab_size`) werden über Fallback-Methoden ermittelt

---

### `vram-calculator.py` — Terminal-Rechner

Liest `models_cache.json` und gibt für jedes LLM-Modell eine VRAM-Matrix im Terminal aus.

**Aufruf:**

```bash
python vram-calculator.py
```

**Konfiguration** (direkt im Skript):

```python
GPU_LIMITS = {
    "6GB (Entry)":  6.0,
    "12GB (Mid)":   12.0,
    "16GB (Pro)":   16.0,
    "24GB (Ultra)": 24.0,
}

USECASES = {
    "Chat (8k)":    8000,
    "Code (32k)":   32000,
    "Doc (64k)":    64000,
    "Rev (128k)":   128000,
    "Res (256k)":   256000,
    "Agent (512k)": 512000,
    "Agent (1M)":   1000000,
}
```

**VRAM-Formel:**

```
Gewichte-VRAM  = file_size_gb
KV-Cache-VRAM  = (2 × n_layers × n_kv_heads × head_dim × ctx_tokens × 2) / 1024³
Gesamt         = Gewichte + KV-Cache
```

`head_dim = n_embd / n_heads`

Bei SSM-Modellen (`n_kv_heads = 0`) entfällt der KV-Cache-Term.

**Farbcodierung:**

| Symbol | Bedeutung |
|---|---|
| 🟢 | Passt bequem (≤ 85 % der GPU-Kapazität) |
| 🟡 | Passt knapp (≤ 100 %) |
| 🔴 | Passt nicht |

---

### `index.html` — Interaktive Webansicht

Eine reine Browser-Anwendung ohne Build-Schritt. Sie besteht aus drei Dateien:

| Datei | Aufgabe |
|---|---|
| `index.html` | Struktur: Tabelle, Filter-Controls |
| `filter.js` | Logik: JSON laden, VRAM berechnen, filtern, sortieren |
| `style.css` | Dark-Mode-Design |

**Funktionen:**

- **Kontext-Auswahl** — wechselt die angezeigte VRAM-Spalte (Chat 8k bis Agent 1M)
- **Architektur-Filter** — zeigt nur Modelle einer bestimmten Architektur
- **Quantisierungs-Filter** — filtert nach Quant-Typ
- **Suche** — Freitext-Suche über Modellname und Architektur
- **Rote ausblenden** — vier Checkboxen (6 / 12 / 16 / 24 GB): blendet Modelle aus, die für die jeweilige GPU-Größe zu groß sind
- **Sortierung** — jede Spalte ist klickbar, auf- und absteigend

**GPU-Fit-Zellen:**

| Symbol | CSS-Klasse | Bedeutung |
|---|---|---|
| `✓` | `fit-good` | Passt bequem (≤ 85 %) |
| `~` | `fit-tight` | Passt knapp (≤ 100 %) |
| `✗` | `fit-none` | Passt nicht |

Der Tooltip am Modellnamen warnt, wenn der gewählte Kontext das ursprüngliche Trainings-Kontextfenster des Modells überschreitet.

`filter.js` lädt `models_cache.json` per `fetch()` und führt dieselbe VRAM-Berechnung wie `vram-calculator.py` im Browser durch — kein Server-Rendering, kein Build-Prozess.

---

## Projektstruktur

```
vram-model-calculator/
├── gguf-scanner.py       # Schritt 1: GGUF-Dateien scannen → models_cache.json
├── vram-calculator.py    # Schritt 2 (optional): Terminal-Ausgabe
├── models_cache.json     # Generiert vom Scanner
├── index.html            # Browser-UI
├── filter.js             # Logik der Browser-UI
├── style.css             # Styles der Browser-UI
└── pyproject.toml        # Python-Abhängigkeiten (gguf, tqdm)
```
