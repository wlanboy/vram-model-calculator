# Verbesserungsplan VRAM Model Calculator

Stand: 2026-11-19. Grundlage ist der Code auf `main` (Commit `036c44d`).

Die Punkte sind nach Nutzen sortiert. Jeder Punkt nennt das Problem, die betroffenen Stellen und einen Vorschlag.
Aufwand: S = unter einer Stunde, M = ein halber Tag, L = mehr als ein Tag.

---

## 1. Genauigkeit der VRAM-Berechnung

Die Frage „Passt das Modell?“ beantwortet das Tool heute mit `Gewichte + KV-Cache (+ Recurrent-State)`. In der
Praxis belegt llama.cpp mehr. Deshalb zeigt die Tabelle Modelle als grün an, die auf der Karte dann nicht laden.

### 1.1 Compute-Buffer und Laufzeit-Overhead fehlen (M)

llama.cpp reserviert zusätzlich einen Compute-Buffer. Er hängt vor allem von `n_ubatch`, `n_embd`,
`n_vocab` (Logits) und dem Kontext ab, bei großen Vokabularen und langem Kontext sind das schnell 0,5 bis 2 GB. Dazu
kommt der Treiber-Kontext (CUDA/SYCL/Vulkan, grob 0,3 bis 0,5 GB). Die Schwelle `TIGHT_FIT_RATIO = 0.85` fängt das
nur pauschal ab und passt bei kleinen Karten gar nicht.

**Vorschlag.** Beim Scan eine Schätzung `compute_buffer_bytes(ctx, n_ubatch)` aus den vorhandenen Feldern
(`vocab_size`, `n_embd`, `n_ff`) berechnen. Im Web und in der CLI einen festen Overhead-Wert einstellbar machen
(Default z. B. 0,5 GB). Die Formel gegen echte llama.cpp-Logs (`llama_context: … compute buffer size`) für drei,
vier Modelle kalibrieren und die Messwerte als Testfälle ablegen.

**Betroffen.** `_model.py`, `vram_calculator.py`, `filter.js`, README-Abschnitt „VRAM-Formel“.

### 1.5 Bedeutung von „Sessions“ klarstellen (S)

Das Tool rechnet `Kontext × Sessions`. In llama.cpp teilt `-np N` den Kontext `-c` auf die Slots
auf, bei `-c 32768 -np 4` hat jeder Slot nur 8k. Wer die Werte 1:1 nach llama.cpp überträgt, liest sie leicht
falsch.

**Vorschlag.** Neben dem Select steht ein Hinweis („entspricht `-c <ctx×N> -np N`“), derselbe Satz kommt ins
README. Optional zeigt die Seite den fertigen llama-server-Aufruf für die aktuelle Auswahl (siehe 3.3).

---

## 2. CLI und Paketstruktur

### 2.1 `vram-calculator` funktioniert nur im Repo-Verzeichnis (S)

`CACHE_FILE = "models_cache.json"` ist relativ zum aktuellen Verzeichnis, ebenso
`model-metadata.txt`. Das installierte Tool (`uv tool install`) findet den Cache in einem anderen Verzeichnis
nicht.

**Vorschlag.** Scanner, Rechner und Checker bekommen die Option `--cache PATH` und lesen die Umgebungsvariable
`VRAM_CALC_CACHE`.
Default bleibt das aktuelle Verzeichnis, damit die GitHub-Pages-Seite weiter funktioniert.

### 2.2 Ein Einstiegspunkt statt `main.py` plus drei Skripte (S bis M)

`main.py` liegt außerhalb des Pakets und landet nicht im Wheel. `hf-copy` gibt es nur dort.
`main.py` überschreibt `sys.argv`, um `gguf_checker.main()` und `hf_cache_picker.main()` aufzurufen, und
`hf_cache_picker` parst seine Argumente von Hand.

**Vorschlag.** `main.py` als `vram_model_calculator/cli.py` ins Paket ziehen und in `pyproject.toml` als
`vram-model-calculator = "vram_model_calculator.cli:main"` eintragen. Alle `main()`-Funktionen bekommen
`main(argv=None)` mit `argparse`, wie `vram_calculator.main` es schon macht. Die drei alten Kommandos bleiben als
Aliase.

### 2.3 CLI-Ausgabe filterbar und konfigurierbar machen (M)

`calculate` gibt alle Modelle mit festen vier GPUs und sieben Usecases aus. Die Werte stehen als
Konstanten im Skript, und die Usecases rechnen mit `8000` statt `8192` Token, während das Web mit 8192 startet.

**Vorschlag.** Die CLI bekommt diese Optionen:

- `--gpu 12 --gpu 24` statt `GPU_LIMITS`, `--ctx 8k,32k,128k` statt `USECASES` (Zweierpotenzen als Default)
- `--filter qwen` (Name/Architektur), `--fits 24` (nur Modelle, die in 24 GB passen), `--sessions N`
- `--json` für Skripte
- eine kompakte Tabelle mit einer Zeile pro Modell als Alternative zur Matrix je Modell

### 2.4 `rescan.sh` durch eine Option ersetzen (S)

Das Skript ruft `python3` direkt auf und setzt `file_size_bytes` auf `0`, um einen Rescan zu
erzwingen. Das ist ein Umweg über ein Datenfeld.

**Vorschlag.** Der Scanner bekommt `scan --rescan-all` (und `--rescan PATTERN`), `rescan.sh` entfällt.

### 2.5 Fest eingebaute Pfade und Filter konfigurierbar machen (S bis M)

`DEFAULT_DIRS` (inkl. `/wdblack/models`), `DEFAULT_DEST` im HF-Picker und `SKIP_PATH_SUBSTRINGS`
stehen fest im Code. Für andere Nutzer der öffentlichen Seite passt das nicht, und der Namensfilter (`dolphin`,
`huihui` …) lässt sich nicht abschalten.

**Vorschlag.** Optionale Konfigurationsdatei `~/.config/vram-model-calculator/config.toml` (über `tomllib`, ohne
neue Abhängigkeit) mit `scan_dirs`, `copy_dest`, `skip_patterns`. Zusätzlich `scan --no-skip`.

### 2.6 Doppelte Dateisuche zwischen Scanner und Checker (S)

`_collect_gguf_files` gibt es zweimal. Der Scanner vergleicht die Endung mit `.lower()`, der Checker
nicht. `MODEL.GGUF` wird also gescannt, aber nicht geprüft.

**Vorschlag.** Eine gemeinsame Funktion in einem kleinen Modul (z. B. `files.py`), die auch die Shard-Gruppierung
liefert. Beide Tools nutzen sie.

### 2.7 Legacy-Code im Rechner aufräumen (S)

`coerce_legacy_cache_int` liest Werte, die als einzelne Zeichen gespeichert waren (`'('` = 40). Der
Cache steht inzwischen bei Version 44 und wird beim Scan migriert.

**Vorschlag.** Die Migration einmal im Scanner erledigen (wie `_backfill_kv_bytes`) und im Rechner nur noch
`int`-Werte erwarten. Das vereinfacht auch den Abgleich mit `filter.js`, das diese Sonderfälle nicht kennt.

---

## 3. Webansicht

### 3.1 Einstellungen in der URL halten (S)

Kontext, Sessions, KV-Typ, Filter und Sortierung gehen beim Neuladen verloren und lassen sich nicht
verlinken.

**Vorschlag.** `state` in `location.hash` serialisieren (z. B. `#ctx=32768&s=2&kv=q8_0&gpu=16,24`) und beim
Laden auslesen.

### 3.2 Eigene GPU-Größe und Multi-GPU (S bis M)

Es gibt nur die zehn festen Pills von 4 bis 80 GB. Karten mit 10, 20 oder 96 GB fehlen, ebenso
2× 24 GB oder Unified Memory mit nutzbarem Anteil (EVO-X2 mit 96 von 128 GB).

**Vorschlag.** Ein Eingabefeld „eigene Größe“, das eine weitere Pill anlegt. Bei Multi-GPU zusätzlich einen
Puffer pro Karte abziehen (Compute-Buffer liegt auf jeder Karte).

### 3.3 Aufschlüsselung pro Zeile (S)

Die Spalte „Gesamt“ zeigt eine Zahl, aber nicht, woraus sie besteht.

**Vorschlag.** Tooltip oder aufklappbare Zeile mit Gewichte / KV-Cache / Recurrent-State / Overhead / mmproj und
der passenden llama-server-Zeile (`-c`, `-np`, `--cache-type-k/-v`, `-fa`, ggf. `--n-cpu-moe`).

### 3.4 Quantisierungen eines Modells gruppieren (M)

Dasselbe Modell in Q4_K_M, Q6_K und Q8_0 steht als drei unabhängige Zeilen in der Tabelle.

**Vorschlag.** Optionale Gruppierung nach `name`/`size_label`. Die erste Zeile zeigt die größte Quantisierung, die
noch in die gewählte GPU passt.

### 3.5 Kleinere UI-Punkte (S)

- Label „filter red“ ist englisch, der Rest der Seite deutsch.
- Untertitel nennt nur „Modellgewichte + KV-Cache“; nach 1.1/1.2 anpassen.
- Sortierbare `th` sind nicht per Tastatur bedienbar und haben kein `aria-sort`.
- `fetch("models_cache.json")` umgeht den Browser-Cache nicht. Nach einem Push zeigt die Seite deshalb manchmal noch
  den alten Stand. `{ cache: "no-cache" }` oder `?v=<_version>` behebt das.
- Die Modellfamilien-Liste in der Navigation ist in `index.html` hart codiert und muss bei jeder neuen Seite von Hand
  ergänzt werden.

---

## 4. Hardware-Seiten

### 4.1 Daten aus dem Cache erzeugen statt von Hand pflegen (L)

Die `MODELS`-Arrays in `hardware/*/index.html` sind statisch und unabhängig von `models_cache.json`.
VRAM-Werte, Kontext und Status können vom Rechner abweichen, sobald sich Formel oder Cache ändern.

**Vorschlag.** Pro Gerät eine kleine Profildatei (`hardware/<Gerät>/device.json`: VRAM, nutzbarer Anteil,
Bandbreite, gemessene t/s). Ein Generator (`main.py hardware`) rechnet die Tabelle mit derselben Logik wie
`vram_calculator.py` und schreibt das `MODELS`-Array. Gemessene Werte (t/s, Bewertung) bleiben Handarbeit in der
Profildatei.

### 4.2 Escaping in `hardware.js` (S)

`hardware.js` setzt Namen und Typen ohne Escaping per `innerHTML` ein. Heute sind die Daten selbst
geschrieben, mit 4.1 kämen sie aus GGUF-Metadaten.

**Vorschlag.** Dieselbe `esc()`-Funktion wie in `filter.js` verwenden, am besten aus einer gemeinsamen Datei.

---

## 6. Doku und Kleinkram

- README nennt den CI-Pfad `.github/workflow/ci.yml` (richtig: `workflows`) und listet den pytest-Schritt nicht.
- README empfiehlt `pip install -r requirements.txt` und `python -m http.server`. Beides sollte einheitlich über
  `uv` laufen (`uv run python -m http.server 8000`).
- `.gitignore` enthält nur `__pycache__/`. Es fehlen `model-metadata.txt`, `models_cache.json.tmp`, `dist/`,
  `.venv/`, `.pytest_cache/`, `.ruff_cache/`.
- Der README-Abschnitt „Konfiguration (direkt im Skript)“ entfällt nach 2.3.

---

## Vorgeschlagene Reihenfolge

1. **Schnelle Fixes.** 6, 2.1, 2.4, 2.6, 3.5, 4.2. Wenig Aufwand, behebt echte Stolpersteine.
2. **Genauigkeit.** 1.1, 1.2, 1.5. Das bringt für die Kernfrage am meisten.
3. **Absicherung.** 5.1, 5.2, bevor die Formel weiter wächst.
4. **Funktionen.** 1.3, 1.4, 2.2, 2.3, 3.1 bis 3.4.
5. **Generator für Hardware-Seiten.** 4.1, sobald 1.x steht, damit die Seiten gleich mit der neuen Formel entstehen.
