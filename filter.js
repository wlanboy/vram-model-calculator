(function () {
    "use strict";

    const { esc, initSortableHeaders, markSorted } = window.VramCommon;

    const GPU_SIZES = [4, 6, 8, 12, 16, 24, 32, 48, 64, 80];
    const GPU_COUNTS = [1, 2, 3, 4];
    const SESSIONS_MAX = 20;

    const CTX_MIN     = 1024;
    const CTX_STEP    = 1024;
    const CTX_DEFAULT = 8192;

    const MODEL_TYPE_LLM = "llm";
    const MODEL_TYPE_MMPROJ = "mmproj";
    const GIB = 1024 ** 3;

    // A GPU is "tight" once usage crosses this fraction of its VRAM. Must match
    // vram_calculator.py (checked by tests/test_vram_calculator.py).
    const TIGHT_FIT_RATIO = 0.85;

    // KV-Cache-Typen wie bei llama.cpp --cache-type-k/--cache-type-v. Der
    // Cache speichert f16-Bytes; q8_0 = 34, q4_0 = 18 Bytes pro 32 Elemente
    // statt 64. Muss mit KV_CACHE_TYPES in vram_calculator.py übereinstimmen
    // (geprüft von tests/test_vram_calculator.py).
    const KV_CACHE_TYPES = {
        f16:  1,
        q8_0: 34 / 64,
        q4_0: 18 / 64,
    };
    const KV_CACHE_TYPE_DEFAULT = "f16";

    // Treiber-/Laufzeit-Overhead pro GPU in GB. Muss mit DEFAULT_OVERHEAD_GB
    // in vram_calculator.py übereinstimmen.
    const DEFAULT_OVERHEAD_GB = 0.5;

    // KQ-Maske im Compute-Buffer: f16 pro KV-Zelle und Ubatch-Token (512).
    // Muss mit COMPUTE_MASK_BYTES_PER_CTX_TOKEN in _model.py übereinstimmen.
    const COMPUTE_MASK_BYTES_PER_CTX_TOKEN = 1024;

    const DEFAULT_VISIBLE = ["6", "12"];

    let models = [];
    let mmprojByDir = {};
    const expanded = new Set();

    function defaultState() {
        return {
            ctx:       CTX_DEFAULT,
            sessions:  1,
            kvType:    KV_CACHE_TYPE_DEFAULT,
            gpuCount:  1,
            overheadGb: DEFAULT_OVERHEAD_GB,
            mcp:       false,
            thinking:  false,
            filterRed: false,
            search:    "",
            visible:   new Set(DEFAULT_VISIBLE),
            customGpus: [],
            groupGb:   0,
            sortCol:   "name",
            sortDir:   1,
        };
    }

    let state = defaultState();

    // ── URL-Zustand ──────────────────────────────────────────
    // Alle Einstellungen stehen im Hash, z. B. #ctx=32768&s=2&kv=q8_0&gpu=16,24.
    // Werte, die dem Default entsprechen, fehlen im Hash.

    function numList(text) {
        return String(text || "").split(",").map(Number).filter(n => n > 0 && Number.isFinite(n));
    }

    function stateToHash() {
        const d = defaultState();
        const p = new URLSearchParams();
        if (state.ctx !== d.ctx) p.set("ctx", state.ctx);
        if (state.sessions !== d.sessions) p.set("s", state.sessions);
        if (state.kvType !== d.kvType) p.set("kv", state.kvType);
        if (state.gpuCount !== d.gpuCount) p.set("n", state.gpuCount);
        if (state.overheadGb !== d.overheadGb) p.set("oh", state.overheadGb);
        const visible = Array.from(state.visible).map(Number).sort((a, b) => a - b).join(",");
        if (visible !== DEFAULT_VISIBLE.join(",")) p.set("gpu", visible);
        if (state.customGpus.length) p.set("cgpu", state.customGpus.join(","));
        if (state.groupGb) p.set("grp", state.groupGb);
        if (state.search) p.set("q", state.search);
        if (state.mcp) p.set("mcp", "1");
        if (state.thinking) p.set("think", "1");
        if (state.filterRed) p.set("red", "1");
        if (state.sortCol !== d.sortCol || state.sortDir !== d.sortDir) {
            p.set("sort", state.sortCol + (state.sortDir === 1 ? "" : ",desc"));
        }
        return p.toString();
    }

    function hashToState(hash) {
        const p = new URLSearchParams(hash.replace(/^#/, ""));
        const s = defaultState();
        const ctx = Number(p.get("ctx"));
        if (ctx >= CTX_MIN) s.ctx = Math.round(ctx / CTX_STEP) * CTX_STEP;
        const sessions = Number(p.get("s"));
        if (sessions >= 1 && sessions <= SESSIONS_MAX) s.sessions = Math.floor(sessions);
        if (p.get("kv") in KV_CACHE_TYPES) s.kvType = p.get("kv");
        const n = Number(p.get("n"));
        if (GPU_COUNTS.includes(n)) s.gpuCount = n;
        const oh = Number(p.get("oh"));
        if (p.has("oh") && oh >= 0 && Number.isFinite(oh)) s.overheadGb = oh;
        s.customGpus = numList(p.get("cgpu"));
        if (p.has("gpu")) s.visible = new Set(numList(p.get("gpu")).map(String));
        s.groupGb = numList(p.get("grp"))[0] || 0;
        s.search = p.get("q") || "";
        s.mcp = p.get("mcp") === "1";
        s.thinking = p.get("think") === "1";
        s.filterRed = p.get("red") === "1";
        const sort = (p.get("sort") || "").split(",");
        if (sort[0] in COLUMN_ACCESSORS) {
            s.sortCol = sort[0];
            s.sortDir = sort[1] === "desc" ? -1 : 1;
        }
        return s;
    }

    function saveState() {
        const hash = stateToHash();
        const url = location.pathname + location.search + (hash ? "#" + hash : "");
        history.replaceState(null, "", url);
    }

    // ── VRAM calculation ─────────────────────────────────────

    // Bytes/token kommt vorberechnet aus models_cache.json (siehe
    // ModelShape.kv_bytes_per_ctx_token/_swa in _model.py) - einzige Quelle
    // der KV-Cache-Formel. kvBytesPerCtxTokenSwa gilt nur bis swaWindow Tokens
    // (Sliding-Window-Attention-Layer wie bei Gemma3/4, Cohere2, gpt-oss
    // cachen nie mehr als das Fenster, unabhängig vom gewählten Kontext).
    // Die Bytes gelten für f16, state.kvType skaliert auf q8_0/q4_0 um.
    function calcKv(model) {
        let bytes = model.kvBytesPerCtxToken * state.ctx;
        if (model.swaWindow) {
            bytes += model.kvBytesPerCtxTokenSwa * Math.min(state.ctx, model.swaWindow);
        }
        return bytes * KV_CACHE_TYPES[state.kvType] / GIB;
    }

    // Recurrent-State von SSM- und Hybrid-Modellen: fest pro Session, in
    // llama.cpp immer f32, darum ohne KV-Cache-Typ.
    function calcState(model) {
        return model.recurrentStateBytes / GIB;
    }

    // Speicher, den jede Session belegt: KV-Cache plus Recurrent-State.
    function calcPerSession(model) {
        return calcKv(model) + calcState(model);
    }

    // Compute-Buffer (siehe compute_buffer_bytes in _model.py) plus KQ-Maske
    // über den Gesamtkontext aller Sessions (-c = ctx × Sessions). Liegt auf
    // jeder GPU einmal.
    function calcCompute(model) {
        const ctxTotal = state.ctx * state.sessions;
        return (model.computeBufferBytes + ctxTotal * COMPUTE_MASK_BYTES_PER_CTX_TOKEN) / GIB;
    }

    function calcPerGpuFixed(model) {
        return calcCompute(model) + state.overheadGb;
    }

    function effectiveTotal(model) {
        return model.size_gb + calcPerSession(model) * state.sessions + state.gpuCount * calcPerGpuFixed(model);
    }

    function capacity(gb) {
        return gb * state.gpuCount;
    }

    function buildModels(raw) {
        const result = [];
        mmprojByDir = {};
        for (const [key, data] of Object.entries(raw)) {
            if (typeof data !== "object" || data === null) continue;
            if (data.type === MODEL_TYPE_MMPROJ) {
                const dir = key.includes("/") ? key.slice(0, key.lastIndexOf("/")) : "";
                if (!(dir in mmprojByDir)) mmprojByDir[dir] = data.file_size_gb || 0;
                continue;
            }
            if (data.type !== MODEL_TYPE_LLM) continue;
            const moe = (data.n_experts && data.n_experts_used)
                ? data.n_experts_used + "/" + data.n_experts
                : null;
            // null oder 0 = reines SSM ohne KV-Cache, wie is_pure_ssm in _model.py
            const kvHeads = data.n_kv_heads;
            const isSSM = kvHeads === null || kvHeads === undefined || kvHeads === 0;
            result.push({
                key,
                dir:      key.includes("/") ? key.slice(0, key.lastIndexOf("/")) : "",
                name:     data.name || key,
                arch:     data.arch || "unknown",
                quant:    data.quant || null,
                sizeLabel: data.size_label || "",
                size_gb:  data.file_size_gb || 0,
                n_ctx_orig: data.n_ctx_orig || null,
                mcp:      !!data.mcp,
                thinking: !!data.thinking,
                moe,
                isSSM,
                kvBytesPerCtxToken: data.kv_bytes_per_ctx_token || 0,
                kvBytesPerCtxTokenSwa: data.kv_bytes_per_ctx_token_swa || 0,
                swaWindow: data.swa_window || null,
                recurrentStateBytes: data.recurrent_state_bytes || 0,
                computeBufferBytes: data.compute_buffer_bytes || 0,
            });
        }
        return result;
    }

    // ── Helpers ──────────────────────────────────────────────

    function fmtCtx(tokens) {
        if (!tokens) return "—";
        if (tokens >= 1000000) return (Math.round(tokens / 100000) / 10) + "M";
        return Math.round(tokens / 1000) + "k";
    }

    function fmtGb(gb) {
        return gb.toFixed(2) + " GB";
    }

    function fitClass(total, limitGb) {
        if (total <= limitGb * TIGHT_FIT_RATIO) return "fit-good";
        if (total <= limitGb)                   return "fit-tight";
        return "fit-none";
    }

    const FIT_ICONS = { "fit-good": "✓", "fit-tight": "~", "fit-none": "✗" };

    function gpuSizes() {
        return Array.from(new Set(GPU_SIZES.concat(state.customGpus))).sort((a, b) => a - b);
    }

    function activeGpuSizes() {
        return gpuSizes().filter(gb => state.visible.has(String(gb)));
    }

    function gpuLabel(gb) {
        return (state.gpuCount > 1 ? state.gpuCount + "× " : "") + gb + " GB";
    }

    function isHiddenByRed(model) {
        if (!state.filterRed) return false;
        const total = effectiveTotal(model);
        return activeGpuSizes().some(gb => fitClass(total, capacity(gb)) === "fit-none");
    }

    const COLUMN_ACCESSORS = {
        name:  m => m.name.toLowerCase(),
        arch:  m => m.arch,
        quant: m => m.quant || "",
        size:  m => m.size_gb,
        kv:    m => calcPerSession(m) * state.sessions,
        ctx:   m => m.n_ctx_orig || 0,
        total: m => effectiveTotal(m),
    };

    function colValue(model, col) {
        return (COLUMN_ACCESSORS[col] || COLUMN_ACCESSORS.name)(model);
    }

    function llamaServerCommand(m) {
        const parts = ["llama-server", "-m", m.key, "-c", String(state.ctx * state.sessions)];
        if (state.sessions > 1) parts.push("-np", String(state.sessions));
        parts.push("-ngl", "99", "-fa", "on");
        if (state.kvType !== "f16") {
            parts.push("--cache-type-k", state.kvType, "--cache-type-v", state.kvType);
        }
        return parts.join(" ");
    }

    // ── Gruppierung ──────────────────────────────────────────
    // Fasst Quantisierungen desselben Modells (Name + Größenangabe) zusammen.
    // Gezeigt wird die größte Quantisierung, die in die Ziel-GPU passt, sonst
    // die kleinste.

    function groupRows(rows) {
        const groups = new Map();
        rows.forEach(function (m) {
            const k = m.name.toLowerCase() + "|" + m.sizeLabel.toLowerCase() + "|" + m.arch;
            if (!groups.has(k)) groups.set(k, []);
            groups.get(k).push(m);
        });
        const limit = capacity(state.groupGb);
        return Array.from(groups.values()).map(function (members) {
            const fitting = members.filter(m => effectiveTotal(m) <= limit);
            const pool = fitting.length ? fitting : members;
            const pick = pool.reduce((best, m) =>
                (fitting.length ? m.size_gb > best.size_gb : m.size_gb < best.size_gb) ? m : best);
            return { model: pick, others: members.filter(m => m !== pick) };
        });
    }

    // ── Render ───────────────────────────────────────────────

    function renderHeader() {
        const row = document.getElementById("model-thead-row");
        row.querySelectorAll("th.fit-cell").forEach(th => th.remove());
        activeGpuSizes().forEach(function (gb) {
            const th = document.createElement("th");
            th.className = "fit-cell";
            th.textContent = gpuLabel(gb);
            row.appendChild(th);
        });
        const active = row.querySelector('th[data-col="' + state.sortCol + '"]');
        markSorted(active, state.sortDir);
    }

    function renderSessionsHint() {
        const hint = document.getElementById("sessions-hint");
        hint.textContent = state.sessions > 1
            ? "entspricht -c " + state.ctx * state.sessions + " -np " + state.sessions
            : "entspricht -c " + state.ctx;
    }

    function renderDetail(m, colspan) {
        const kv = calcKv(m);
        const st = calcState(m);
        const compute = calcCompute(m);
        const n = state.gpuCount;
        const perGpu = n > 1 ? " × " + n + " GPUs" : "";
        const sess = state.sessions > 1 ? " × " + state.sessions + " Sessions" : "";
        const items = [
            ["Gewichte", fmtGb(m.size_gb)],
            ["KV-Cache (" + state.kvType + ")", fmtGb(kv) + sess],
        ];
        if (st) items.push(["Recurrent-State", fmtGb(st) + sess]);
        items.push(["Compute-Buffer", fmtGb(compute) + perGpu]);
        items.push(["Overhead", fmtGb(state.overheadGb) + perGpu]);
        items.push(["Gesamt", fmtGb(effectiveTotal(m))]);
        if (m.dir in mmprojByDir) {
            items.push(["mmproj (nur mit --mmproj, nicht im Gesamt)", fmtGb(mmprojByDir[m.dir])]);
        }
        const list = items.map(it => "<dt>" + esc(it[0]) + "</dt><dd>" + esc(it[1]) + "</dd>").join("");
        const moeHint = m.moe
            ? '<p class="detail-hint">MoE: Reicht der VRAM nicht, lagert <code>--n-cpu-moe N</code> die Experten der ersten N Layer auf die CPU aus.</p>'
            : "";
        return '<tr class="detail-row"><td colspan="' + colspan + '">' +
            '<dl class="breakdown">' + list + "</dl>" +
            '<pre class="cmd">' + esc(llamaServerCommand(m)) + "</pre>" + moeHint +
            "</td></tr>";
    }

    function renderRow(m, others, activeGpu, colspan) {
        const kv     = calcPerSession(m);
        const kvEff  = kv * state.sessions;
        const totalEff = effectiveTotal(m);
        const fixed  = state.gpuCount * calcPerGpuFixed(m);
        const moe    = m.moe ? '<span class="badge badge-moe">MoE ' + esc(m.moe) + '</span>' : "";
        const feats  =
            (m.mcp      ? '<span class="feat-icon" title="Unterstützt Tool Calls (MCP)">🔧</span>' : "") +
            (m.thinking ? '<span class="feat-icon" title="Unterstützt Thinking/Reasoning">🧠</span>' : "");
        const ctxOver = m.n_ctx_orig && state.ctx > m.n_ctx_orig
            ? ' <span class="badge badge-warn" title="Über Trainings-Kontextfenster (' + esc(m.n_ctx_orig.toLocaleString()) + ' Token)">ctx</span>'
            : "";
        const group = others.length
            ? ' <span class="badge badge-group" title="' + esc("Weitere Quants: " + others.map(o => (o.quant || "?") + " " + o.size_gb.toFixed(1) + " GB").join(", ")) + '">+' + others.length + "</span>"
            : "";
        const gpuCells = activeGpu.map(function (gb) {
            const cap  = capacity(gb);
            const cls  = fitClass(totalEff, cap);
            let parallelHtml = "";
            // Auch reine SSMs belegen pro Session Speicher (Recurrent-State).
            // Fehlt der Wert (Cache-Eintrag vor dem Rescan), bleibt die
            // Zelle leer statt einer unbegrenzten Zahl.
            if (cls !== "fit-none" && kv > 0) {
                const remaining = cap - m.size_gb - fixed;
                const parallel = remaining > 0 ? Math.floor(remaining / kv) : 0;
                if (parallel > 0) {
                    parallelHtml = '<span class="parallel-count">' + parallel + "×</span>";
                }
            }
            return '<td class="fit-cell ' + cls + '">' + FIT_ICONS[cls] + (parallelHtml ? "<br>" + parallelHtml : "") + "</td>";
        }).join("");

        const kvDisplay =
            (m.isSSM ? '<span class="badge badge-arch">SSM</span> ' : "") +
            kvEff.toFixed(2) + " GB" + (state.sessions > 1 ? ' <span class="col-muted">×' + state.sessions + '</span>' : "");
        const kvTitle = m.recurrentStateBytes
            ? ' title="Enthält ' + (m.recurrentStateBytes / (1024 ** 2)).toFixed(1) + ' MB Recurrent-State pro Session"'
            : "";
        const isOpen = expanded.has(m.key);

        const row = [
            "<tr>",
            '<td class="col-name" title="' + esc(m.key) + '">' +
                '<button type="button" class="row-toggle" data-key="' + esc(m.key) + '" aria-expanded="' + isOpen + '" aria-label="Aufschlüsselung ' + (isOpen ? "schließen" : "öffnen") + '">' + (isOpen ? "▾" : "▸") + "</button>" +
                esc(m.name) + feats + ctxOver + group + "</td>",
            "<td><span class='badge badge-arch'>" + esc(m.arch) + "</span>" + moe + "</td>",
            "<td>" + (m.quant ? esc(m.quant) : "—") + "</td>",
            '<td class="col-mono col-muted">' + m.size_gb.toFixed(2) + " GB</td>",
            '<td class="col-mono"' + kvTitle + ">" + kvDisplay + "</td>",
            '<td class="col-mono col-muted">' + fmtCtx(m.n_ctx_orig) + "</td>",
            '<td class="col-mono" title="davon ' + fixed.toFixed(2) + ' GB Compute-Buffer und Overhead">' + totalEff.toFixed(2) + " GB</td>",
            gpuCells,
            "</tr>",
        ].join("");
        return row + (isOpen ? renderDetail(m, colspan) : "");
    }

    function renderTable() {
        saveState();
        renderHeader();
        renderSessionsHint();
        const tbody = document.getElementById("model-tbody");
        const q = state.search.toLowerCase();

        let rows = models.filter(m => {
            if (state.mcp      && !m.mcp)      return false;
            if (state.thinking && !m.thinking) return false;
            if (q && !m.name.toLowerCase().includes(q) && !m.arch.toLowerCase().includes(q)) return false;
            if (isHiddenByRed(m)) return false;
            return true;
        });

        let entries = state.groupGb
            ? groupRows(rows)
            : rows.map(m => ({ model: m, others: [] }));

        entries.sort((a, b) => {
            const va = colValue(a.model, state.sortCol);
            const vb = colValue(b.model, state.sortCol);
            if (va < vb) return -state.sortDir;
            if (va > vb) return  state.sortDir;
            return 0;
        });

        document.getElementById("row-count").textContent =
            entries.length + " von " + models.length + " Modellen" +
            (state.groupGb ? " (" + rows.length + " Quants gruppiert)" : "");

        const activeGpu = activeGpuSizes();
        const totalCols = 7 + activeGpu.length;

        if (entries.length === 0) {
            tbody.innerHTML =
                '<tr><td colspan="' + totalCols + '" class="no-results">Keine Modelle gefunden.</td></tr>';
            return;
        }

        tbody.innerHTML = entries.map(e => renderRow(e.model, e.others, activeGpu, totalCols)).join("");
    }

    // ── Controls ─────────────────────────────────────────────

    function renderPills() {
        const el = document.getElementById("hide-vram-pills");
        el.innerHTML = "";
        gpuSizes().forEach(function (gb) {
            const key = String(gb);
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "pill" + (state.visible.has(key) ? " selected" : "");
            btn.setAttribute("aria-pressed", state.visible.has(key));
            btn.textContent = gb + " GB";
            btn.addEventListener("click", function () {
                if (state.visible.has(key)) state.visible.delete(key); else state.visible.add(key);
                renderPills();
                renderTable();
            });
            el.appendChild(btn);
            if (state.customGpus.includes(gb)) {
                const rm = document.createElement("button");
                rm.type = "button";
                rm.className = "pill pill-remove";
                rm.textContent = "×";
                rm.title = gb + " GB entfernen";
                rm.setAttribute("aria-label", gb + " GB entfernen");
                rm.addEventListener("click", function () {
                    state.customGpus = state.customGpus.filter(g => g !== gb);
                    state.visible.delete(key);
                    if (state.groupGb === gb) state.groupGb = 0;
                    syncControls();
                    renderTable();
                });
                el.appendChild(rm);
            }
        });
    }

    function renderGroupSelect() {
        document.getElementById("group-select").innerHTML =
            '<option value="0">aus</option>' +
            gpuSizes().map(gb => '<option value="' + gb + '"' + (gb === state.groupGb ? " selected" : "") + ">für " + gb + " GB</option>").join("");
    }

    function syncControls() {
        document.getElementById("ctx-input").value = state.ctx;
        document.getElementById("sessions-select").value = String(state.sessions);
        document.getElementById("kv-type-select").value = state.kvType;
        document.getElementById("gpu-count-select").value = String(state.gpuCount);
        document.getElementById("overhead-input").value = state.overheadGb;
        document.getElementById("filter-mcp").checked = state.mcp;
        document.getElementById("filter-thinking").checked = state.thinking;
        document.getElementById("filter-red").checked = state.filterRed;
        document.getElementById("search").value = state.search;
        renderPills();
        renderGroupSelect();
    }

    function populateFilters() {
        document.getElementById("sessions-select").innerHTML =
            Array.from({ length: SESSIONS_MAX }, (_, i) => i + 1)
                .map(n => '<option value="' + n + '">' + n + (n === 1 ? " Session" : " Sessions") + "</option>")
                .join("");

        document.getElementById("kv-type-select").innerHTML =
            Object.keys(KV_CACHE_TYPES).map(t => '<option value="' + t + '">' + t + "</option>").join("");

        document.getElementById("gpu-count-select").innerHTML =
            GPU_COUNTS.map(n => '<option value="' + n + '">' + n + (n === 1 ? " GPU" : " GPUs") + "</option>").join("");

        syncControls();
    }

    function initControls() {
        const ctxInput = document.getElementById("ctx-input");

        function setCtx(value) {
            state.ctx = Math.max(CTX_MIN, Math.round(value / CTX_STEP) * CTX_STEP);
            ctxInput.value = state.ctx;
            renderTable();
        }

        function on(id, event, handler) {
            document.getElementById(id).addEventListener(event, function (e) {
                handler(e.target);
                renderTable();
            });
        }

        document.getElementById("ctx-minus").addEventListener("click", () => setCtx(state.ctx - CTX_STEP));
        document.getElementById("ctx-plus").addEventListener("click", () => setCtx(state.ctx + CTX_STEP));
        ctxInput.addEventListener("change", e => setCtx(Number(e.target.value) || CTX_DEFAULT));

        on("sessions-select", "change", t => { state.sessions = Number(t.value); });
        on("kv-type-select", "change", t => { if (t.value in KV_CACHE_TYPES) state.kvType = t.value; });
        on("gpu-count-select", "change", t => { state.gpuCount = Number(t.value) || 1; });
        on("overhead-input", "change", t => {
            const v = Number(t.value);
            state.overheadGb = v >= 0 && Number.isFinite(v) ? v : DEFAULT_OVERHEAD_GB;
            t.value = state.overheadGb;
        });
        on("group-select", "change", t => { state.groupGb = Number(t.value) || 0; });
        on("filter-mcp", "change", t => { state.mcp = t.checked; });
        on("filter-thinking", "change", t => { state.thinking = t.checked; });
        on("filter-red", "change", t => { state.filterRed = t.checked; });
        on("search", "input", t => { state.search = t.value; });

        document.getElementById("gpu-custom-form").addEventListener("submit", function (e) {
            e.preventDefault();
            const input = document.getElementById("gpu-custom-input");
            const gb = Math.round(Number(input.value) * 10) / 10;
            if (!(gb > 0)) return;
            if (!GPU_SIZES.includes(gb) && !state.customGpus.includes(gb)) state.customGpus.push(gb);
            state.visible.add(String(gb));
            input.value = "";
            syncControls();
            renderTable();
        });

        document.getElementById("model-tbody").addEventListener("click", function (e) {
            const btn = e.target.closest(".row-toggle");
            if (!btn) return;
            const key = btn.dataset.key;
            if (expanded.has(key)) expanded.delete(key); else expanded.add(key);
            renderTable();
        });

        initSortableHeaders(function (th) {
            const col = th.dataset.col;
            if (state.sortCol === col) {
                state.sortDir *= -1;
            } else {
                state.sortCol = col;
                state.sortDir = 1;
            }
            renderTable();
        });

        window.addEventListener("hashchange", function () {
            state = hashToState(location.hash);
            syncControls();
            renderTable();
        });
    }

    // ── Bootstrap ────────────────────────────────────────────

    document.addEventListener("DOMContentLoaded", function () {
        state = hashToState(location.hash);
        // no-cache: nach einem Push sonst manchmal noch der alte Stand.
        fetch("models_cache.json", { cache: "no-cache" })
            .then(function (r) {
                if (!r.ok) throw new Error("HTTP " + r.status);
                return r.json();
            })
            .then(function (raw) {
                models = buildModels(raw);
                populateFilters();
                initControls();
                renderTable();
            })
            .catch(function (err) {
                document.getElementById("model-tbody").innerHTML =
                    '<tr><td colspan="7" class="no-results">Fehler beim Laden: ' + esc(err.message) + "</td></tr>";
            });
    });
}());
