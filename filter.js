(function () {
    "use strict";

    const GPU_LIMITS = [
        { label: "4 GB",  gb: 4  },
        { label: "6 GB",  gb: 6  },
        { label: "8 GB",  gb: 8  },
        { label: "12 GB", gb: 12 },
        { label: "16 GB", gb: 16 },
        { label: "24 GB", gb: 24 },
        { label: "32 GB", gb: 32 },
        { label: "48 GB", gb: 48 },
        { label: "64 GB", gb: 64 },
        { label: "80 GB", gb: 80 },
    ];

    const CTX_MIN     = 1024;
    const CTX_STEP    = 1024;
    const CTX_DEFAULT = 8192;

    const MODEL_TYPE_LLM = "llm";

    // KV-cache dtype is fp16 (2 bytes/element); each layer stores one Key and
    // one Value tensor. Keep these two in sync with the identical constants
    // in vram_calculator.py.
    const KV_BYTES_PER_ELEMENT = 2;
    const KV_TENSORS_PER_LAYER = 2;

    // A GPU is "tight" once usage crosses this fraction of its VRAM. Keep in
    // sync with the identical constant in vram_calculator.py.
    const TIGHT_FIT_RATIO = 0.85;

function activeGpuLimits() {
        return GPU_LIMITS.filter(g => state.hideRed[String(g.gb)]);
    }

    let models = [];

    let state = {
        ctx:      CTX_DEFAULT,
        sessions: 1,
        mcp:       false,
        thinking:  false,
        filterRed: false,
        search:   "",
        hideRed: { "6": true, "12": true },
        sortCol: "name",
        sortDir: 1,
    };

    // ── VRAM calculation ─────────────────────────────────────

    function calcKv(model, ctx) {
        if (model.isSSM || !model.n_kv_heads || !model.n_layers || !model.n_embd) return 0;
        const headDim = Math.floor(model.n_embd / (model.n_heads || 1));
        return (KV_TENSORS_PER_LAYER * model.n_layers * model.n_kv_heads * headDim * ctx * KV_BYTES_PER_ELEMENT) / (1024 ** 3);
    }

    function buildModels(raw) {
        const result = [];
        for (const [key, data] of Object.entries(raw)) {
            if (typeof data !== "object" || data.type !== MODEL_TYPE_LLM) continue;
            const moe = (data.n_experts && data.n_experts_used)
                ? data.n_experts_used + "/" + data.n_experts
                : null;
            const kvHeads = data.n_kv_heads;   // null = SSM/Hybrid (kein KV-Cache)
            const isSSM = kvHeads === null || kvHeads === undefined || kvHeads === 0;
            result.push({
                key,
                name:     data.name || key,
                arch:     data.arch || "unknown",
                quant:    data.quant || null,
                size_gb:  data.file_size_gb || 0,
                n_ctx_orig: data.n_ctx_orig || null,
                mcp:      !!data.mcp,
                thinking: !!data.thinking,
                moe,
                isSSM,
                n_layers:   data.n_layers || 0,
                n_embd:     data.n_embd   || 0,
                n_heads:    data.n_heads  || 1,
                n_kv_heads: kvHeads,
            });
        }
        return result;
    }

    // ── Helpers ──────────────────────────────────────────────

    function fmtCtx(tokens) {
        if (!tokens) return "—";
        if (tokens >= 1000000) return (tokens / 1000000) + "M";
        return Math.round(tokens / 1000) + "k";
    }

    function fitClass(total, limitGb) {
        if (total <= limitGb * TIGHT_FIT_RATIO) return "fit-good";
        if (total <= limitGb)                   return "fit-tight";
        return "fit-none";
    }

    function fitIcon(total, limitGb) {
        if (total <= limitGb * TIGHT_FIT_RATIO) return "✓";
        if (total <= limitGb)                   return "~";
        return "✗";
    }

    function effectiveTotal(model) {
        const kv = calcKv(model, state.ctx);
        return model.size_gb + kv * state.sessions;
    }

    function isHiddenByRed(model) {
        if (!state.filterRed) return false;
        const total = effectiveTotal(model);
        return Object.entries(state.hideRed).some(function (entry) {
            return entry[1] && fitClass(total, Number(entry[0])) === "fit-none";
        });
    }

    const COLUMN_ACCESSORS = {
        name:  m => m.name.toLowerCase(),
        arch:  m => m.arch,
        quant: m => m.quant || "",
        size:  m => m.size_gb,
        kv:    m => calcKv(m, state.ctx) * state.sessions,
        ctx:   m => m.n_ctx_orig || 0,
        total: m => effectiveTotal(m),
    };

    function colValue(model, col) {
        return (COLUMN_ACCESSORS[col] || COLUMN_ACCESSORS.name)(model);
    }

    // ── Render ───────────────────────────────────────────────

    function updateColumnVisibility() {
        document.querySelectorAll("th[data-vram]").forEach(function (th) {
            th.style.display = state.hideRed[th.dataset.vram] ? "" : "none";
        });
    }

    function renderTable() {
        updateColumnVisibility();
        const tbody = document.getElementById("model-tbody");
        const q = state.search.toLowerCase();

        let rows = models.filter(m => {
            if (state.mcp      && !m.mcp)      return false;
            if (state.thinking && !m.thinking) return false;
            if (q && !m.name.toLowerCase().includes(q) && !m.arch.toLowerCase().includes(q)) return false;
            if (isHiddenByRed(m)) return false;
            return true;
        });

        rows.sort((a, b) => {
            const va = colValue(a, state.sortCol);
            const vb = colValue(b, state.sortCol);
            if (va < vb) return -state.sortDir;
            if (va > vb) return  state.sortDir;
            return 0;
        });

        document.getElementById("row-count").textContent =
            rows.length + " von " + models.length + " Modellen";

        const activeGpu = activeGpuLimits();
        const totalCols = 7 + activeGpu.length;

        if (rows.length === 0) {
            tbody.innerHTML =
                '<tr><td colspan="' + totalCols + '" class="no-results">Keine Modelle gefunden.</td></tr>';
            return;
        }

        tbody.innerHTML = rows.map(function (m) {
            const kv     = calcKv(m, state.ctx);
            const kvEff  = kv * state.sessions;
            const totalEff = m.size_gb + kvEff;
            const moe    = m.moe ? '<span class="badge badge-moe">MoE ' + m.moe + '</span>' : "";
            const feats  =
                (m.mcp      ? '<span class="feat-icon" title="Unterstützt Tool Calls (MCP)">🔧</span>' : "") +
                (m.thinking ? '<span class="feat-icon" title="Unterstützt Thinking/Reasoning">🧠</span>' : "");
            const ctxOver = m.n_ctx_orig && state.ctx > m.n_ctx_orig
                ? ' title="Über Trainings-Kontextfenster (' + m.n_ctx_orig.toLocaleString() + ' Token)"'
                : "";
            const gpuCells = activeGpu.map(function (g) {
                const cls  = fitClass(totalEff, g.gb);
                const icon = fitIcon(totalEff, g.gb);
                let parallelHtml = "";
                if (cls !== "fit-none") {
                    if (m.isSSM) {
                        parallelHtml = '<span class="parallel-count">∞×</span>';
                    } else if (kv > 0) {
                        const remaining = g.gb - m.size_gb;
                        const parallel = remaining > 0 ? Math.floor(remaining / kv) : 0;
                        if (parallel > 0) {
                            parallelHtml = '<span class="parallel-count">' + parallel + "×</span>";
                        }
                    }
                }
                return '<td class="fit-cell ' + cls + '">' + icon + (parallelHtml ? "<br>" + parallelHtml : "") + "</td>";
            }).join("");

            const kvDisplay = m.isSSM
                ? '<span class="badge badge-arch">SSM</span>'
                : kvEff.toFixed(2) + " GB" + (state.sessions > 1 ? ' <span class="col-muted">×' + state.sessions + '</span>' : "");

            return [
                "<tr>",
                '<td class="col-name" title="' + m.key + '"' + ctxOver + ">" + m.name + feats + "</td>",
                "<td><span class='badge badge-arch'>" + m.arch + "</span>" + moe + "</td>",
                "<td>" + (m.quant || "—") + "</td>",
                '<td class="col-mono col-muted">' + m.size_gb.toFixed(2) + " GB</td>",
                '<td class="col-mono">' + kvDisplay + "</td>",
                '<td class="col-mono col-muted">' + fmtCtx(m.n_ctx_orig) + "</td>",
                '<td class="col-mono">' + totalEff.toFixed(2) + " GB</td>",
                gpuCells,
                "</tr>",
            ].join("");
        }).join("");
    }

    // ── Filters & sort init ──────────────────────────────────

    function populateFilters() {
        document.getElementById("ctx-input").value = state.ctx;

        document.getElementById("sessions-select").innerHTML =
            Array.from({ length: 20 }, (_, i) => i + 1)
                .map(n => '<option value="' + n + '"' + (n === 1 ? ' selected' : '') + '>' + n + (n === 1 ? ' Session' : ' Sessions') + '</option>')
                .join("");

        const hideVramEl = document.getElementById("hide-vram-pills");
        hideVramEl.innerHTML = "";
        GPU_LIMITS.forEach(function ({ gb }) {
            const btn = document.createElement("button");
            btn.className = "pill" + (state.hideRed[String(gb)] ? " selected" : "");
            btn.dataset.gb = gb;
            btn.textContent = gb + " GB";
            btn.addEventListener("click", function () {
                btn.classList.toggle("selected");
                if (btn.classList.contains("selected")) {
                    state.hideRed[String(gb)] = true;
                } else {
                    delete state.hideRed[String(gb)];
                }
                renderTable();
            });
            hideVramEl.appendChild(btn);
        });
    }

    function initSort() {
        document.querySelectorAll("th.sortable").forEach(function (th) {
            th.addEventListener("click", function () {
                const col = th.dataset.col;
                if (state.sortCol === col) {
                    state.sortDir *= -1;
                } else {
                    state.sortCol = col;
                    state.sortDir = 1;
                }
                document.querySelectorAll("th[data-sorted]").forEach(function (t) {
                    t.removeAttribute("data-sorted");
                });
                th.setAttribute("data-sorted", state.sortDir === 1 ? "asc" : "desc");
                renderTable();
            });
        });
    }

    // ── Bootstrap ────────────────────────────────────────────

    document.addEventListener("DOMContentLoaded", function () {
        fetch("models_cache.json")
            .then(function (r) { return r.json(); })
            .then(function (raw) {
                models = buildModels(raw);
                populateFilters();
                initSort();

                const ctxInput = document.getElementById("ctx-input");

                function setCtx(value) {
                    const clamped = Math.max(CTX_MIN, Math.round(value / CTX_STEP) * CTX_STEP);
                    state.ctx = clamped;
                    ctxInput.value = clamped;
                    renderTable();
                }

                document.getElementById("ctx-minus").addEventListener("click", function () {
                    setCtx(state.ctx - CTX_STEP);
                });
                document.getElementById("ctx-plus").addEventListener("click", function () {
                    setCtx(state.ctx + CTX_STEP);
                });
                ctxInput.addEventListener("change", function (e) {
                    setCtx(Number(e.target.value) || CTX_DEFAULT);
                });
                document.getElementById("sessions-select").addEventListener("change", function (e) {
                    state.sessions = Number(e.target.value); renderTable();
                });
                document.getElementById("filter-mcp").addEventListener("change", function (e) {
                    state.mcp = e.target.checked; renderTable();
                });
                document.getElementById("filter-thinking").addEventListener("change", function (e) {
                    state.thinking = e.target.checked; renderTable();
                });
                document.getElementById("filter-red").addEventListener("change", function (e) {
                    state.filterRed = e.target.checked; renderTable();
                });
                document.getElementById("search").addEventListener("input", function (e) {
                    state.search = e.target.value; renderTable();
                });

                renderTable();
            })
            .catch(function (err) {
                document.getElementById("model-tbody").innerHTML =
                    '<tr><td colspan="' + (7 + GPU_LIMITS.length) + '" class="no-results">Fehler beim Laden: ' + err.message + '</td></tr>';
            });
    });
}());
