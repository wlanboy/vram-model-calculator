(function () {
    "use strict";

    const GPU_LIMITS = [
        { label: "6 GB",  gb: 6  },
        { label: "12 GB", gb: 12 },
        { label: "16 GB", gb: 16 },
        { label: "24 GB", gb: 24 },
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

    const TOTAL_COLS = 6 + GPU_LIMITS.length;

    let models = [];

    let state = {
        ctx:     Object.keys(CTX_TOKENS)[0],
        arch:    "",
        quant:   "",
        search:  "",
        hideRed: {},   // { 6: true, 12: false, … }
        sortCol: "name",
        sortDir: 1,
    };

    // ── VRAM calculation ─────────────────────────────────────

    function calcVram(data) {
        const layers  = data.n_layers    || 0;
        const embd    = data.n_embd      || 0;
        const heads   = data.n_heads     || 1;
        const kvHeads = data.n_kv_heads  || 0;
        const base    = data.file_size_gb || 0;
        const vram    = {};

        for (const [name, ctx] of Object.entries(CTX_TOKENS)) {
            let kv = 0;
            if (kvHeads > 0 && layers > 0 && embd > 0) {
                const headDim = Math.floor(embd / (heads || 1));
                kv = (2 * layers * kvHeads * headDim * ctx * 2) / (1024 ** 3);
            }
            vram[name] = { kv, total: base + kv };
        }
        return vram;
    }

    function buildModels(raw) {
        const result = [];
        for (const [key, data] of Object.entries(raw)) {
            if (typeof data !== "object" || data.type !== "llm") continue;
            const moe = (data.n_experts && data.n_experts_used)
                ? data.n_experts_used + "/" + data.n_experts
                : null;
            result.push({
                key,
                name:       data.name || key,
                arch:       data.arch || "unknown",
                quant:      data.quant || null,
                size_gb:    data.file_size_gb || 0,
                n_ctx_orig: data.n_ctx_orig || null,
                moe,
                vram:       calcVram(data),
            });
        }
        return result;
    }

    // ── Helpers ──────────────────────────────────────────────

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

    function isHiddenByRed(model) {
        const total = (model.vram[state.ctx] || { total: 0 }).total;
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
            case "kv":    return v.kv;
            case "total": return v.total;
            default:      return model.name.toLowerCase();
        }
    }

    // ── Render ───────────────────────────────────────────────

    function renderTable() {
        const tbody = document.getElementById("model-tbody");
        const q = state.search.toLowerCase();

        let rows = models.filter(m => {
            if (state.arch  && m.arch  !== state.arch)  return false;
            if (state.quant && m.quant !== state.quant) return false;
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

        if (rows.length === 0) {
            tbody.innerHTML =
                '<tr><td colspan="' + TOTAL_COLS + '" class="no-results">Keine Modelle gefunden.</td></tr>';
            return;
        }

        tbody.innerHTML = rows.map(function (m) {
            const v      = m.vram[state.ctx] || { kv: 0, total: 0 };
            const moe    = m.moe ? '<span class="badge badge-moe">MoE ' + m.moe + '</span>' : "";
            const ctxOver = m.n_ctx_orig && CTX_TOKENS[state.ctx] > m.n_ctx_orig
                ? ' title="Über Trainings-Kontextfenster (' + m.n_ctx_orig.toLocaleString() + ' Token)"'
                : "";
            const gpuCells = GPU_LIMITS.map(function (g) {
                const cls  = fitClass(v.total, g.gb);
                const icon = fitIcon(v.total, g.gb);
                return '<td class="fit-cell ' + cls + '">' + icon + "</td>";
            }).join("");

            return [
                "<tr>",
                '<td class="col-name" title="' + m.key + '"' + ctxOver + ">" + m.name + "</td>",
                "<td><span class='badge badge-arch'>" + m.arch + "</span>" + moe + "</td>",
                "<td>" + (m.quant || "—") + "</td>",
                '<td class="col-mono col-muted">' + m.size_gb.toFixed(2) + " GB</td>",
                '<td class="col-mono">' + v.kv.toFixed(2)    + " GB</td>",
                '<td class="col-mono">' + v.total.toFixed(2) + " GB</td>",
                gpuCells,
                "</tr>",
            ].join("");
        }).join("");
    }

    // ── Filters & sort init ──────────────────────────────────

    function populateFilters() {
        const archs  = [...new Set(models.map(m => m.arch))].sort();
        const quants = [...new Set(models.map(m => m.quant).filter(Boolean))].sort();

        document.getElementById("arch-select").innerHTML =
            '<option value="">Alle</option>' + archs.map(a => "<option>" + a + "</option>").join("");
        document.getElementById("quant-select").innerHTML =
            '<option value="">Alle</option>' + quants.map(q => "<option>" + q + "</option>").join("");
        document.getElementById("ctx-select").innerHTML =
            Object.keys(CTX_TOKENS).map(l => "<option>" + l + "</option>").join("");

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
                document.getElementById("arch-select").addEventListener("change", function (e) {
                    state.arch = e.target.value; renderTable();
                });
                document.getElementById("quant-select").addEventListener("change", function (e) {
                    state.quant = e.target.value; renderTable();
                });
                document.getElementById("search").addEventListener("input", function (e) {
                    state.search = e.target.value; renderTable();
                });
                document.querySelectorAll("[data-hide-gb]").forEach(function (cb) {
                    cb.addEventListener("change", function (e) {
                        state.hideRed[e.target.dataset.hideGb] = e.target.checked;
                        renderTable();
                    });
                });

                renderTable();
            })
            .catch(function (err) {
                document.getElementById("model-tbody").innerHTML =
                    '<tr><td colspan="' + TOTAL_COLS + '" class="no-results">Fehler beim Laden: ' + err.message + '</td></tr>';
            });
    });
}());
