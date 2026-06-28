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

    const CTX_TOKENS = {
        "Chat (8k)":    8000,
        "Code (32k)":   32000,
        "Doc (64k)":    64000,
        "Rev (128k)":   128000,
        "Res (256k)":   256000,
        "Agent (512k)": 512000,
        "Agent (1M)":   1000000,
    };

function activeGpuLimits() {
        return GPU_LIMITS.filter(g => state.hideRed[String(g.gb)]);
    }

    let models = [];

    let state = {
        ctx:      Object.keys(CTX_TOKENS)[0],
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

    function calcVram(data) {
        const layers  = data.n_layers    || 0;
        const embd    = data.n_embd      || 0;
        const heads   = data.n_heads     || 1;
        const kvHeads = data.n_kv_heads;   // null = SSM/Hybrid (kein KV-Cache)
        const base    = data.file_size_gb || 0;
        const isSSM   = kvHeads === null || kvHeads === 0;
        const vram    = {};

        for (const [name, ctx] of Object.entries(CTX_TOKENS)) {
            let kv = 0;
            if (!isSSM && kvHeads > 0 && layers > 0 && embd > 0) {
                const headDim = Math.floor(embd / (heads || 1));
                kv = (2 * layers * kvHeads * headDim * ctx * 2) / (1024 ** 3);
            }
            vram[name] = { kv, total: base + kv };
        }
        return { vram, isSSM };
    }

    function buildModels(raw) {
        const result = [];
        for (const [key, data] of Object.entries(raw)) {
            if (typeof data !== "object" || data.type !== "llm") continue;
            const moe = (data.n_experts && data.n_experts_used)
                ? data.n_experts_used + "/" + data.n_experts
                : null;
            const { vram, isSSM } = calcVram(data);
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
                vram,
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
        if (total <= limitGb * 0.85) return "fit-good";
        if (total <= limitGb)        return "fit-tight";
        return "fit-none";
    }

    function fitIcon(total, limitGb) {
        if (total <= limitGb * 0.85) return "✓";
        if (total <= limitGb)        return "~";
        return "✗";
    }

    function effectiveTotal(model) {
        const v = model.vram[state.ctx] || { kv: 0, total: 0 };
        return model.size_gb + v.kv * state.sessions;
    }

    function isHiddenByRed(model) {
        if (!state.filterRed) return false;
        const total = effectiveTotal(model);
        return Object.entries(state.hideRed).some(function (entry) {
            return entry[1] && fitClass(total, Number(entry[0])) === "fit-none";
        });
    }

    function colValue(model, col) {
        const v = model.vram[state.ctx] || { kv: 0, total: 0 };
        switch (col) {
            case "name":  return model.name.toLowerCase();
            case "arch":  return model.arch;
            case "quant": return model.quant || "";
            case "size":  return model.size_gb;
            case "kv":    return v.kv * state.sessions;
            case "ctx":   return model.n_ctx_orig || 0;
            case "total": return effectiveTotal(model);
            default:      return model.name.toLowerCase();
        }
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
            const v      = m.vram[state.ctx] || { kv: 0, total: 0 };
            const kvEff  = v.kv * state.sessions;
            const totalEff = m.size_gb + kvEff;
            const moe    = m.moe ? '<span class="badge badge-moe">MoE ' + m.moe + '</span>' : "";
            const ctxOver = m.n_ctx_orig && CTX_TOKENS[state.ctx] > m.n_ctx_orig
                ? ' title="Über Trainings-Kontextfenster (' + m.n_ctx_orig.toLocaleString() + ' Token)"'
                : "";
            const gpuCells = activeGpu.map(function (g) {
                const cls  = fitClass(totalEff, g.gb);
                const icon = fitIcon(totalEff, g.gb);
                let parallelHtml = "";
                if (cls !== "fit-none") {
                    if (m.isSSM) {
                        parallelHtml = '<span class="parallel-count">∞×</span>';
                    } else if (v.kv > 0) {
                        const remaining = g.gb - m.size_gb;
                        const parallel = remaining > 0 ? Math.floor(remaining / v.kv) : 0;
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
                '<td class="col-name" title="' + m.key + '"' + ctxOver + ">" + m.name + "</td>",
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
        document.getElementById("ctx-select").innerHTML =
            Object.keys(CTX_TOKENS).map(l => "<option>" + l + "</option>").join("");

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

        state.ctx = Object.keys(CTX_TOKENS)[0];
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

                document.getElementById("ctx-select").addEventListener("change", function (e) {
                    state.ctx = e.target.value; renderTable();
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
