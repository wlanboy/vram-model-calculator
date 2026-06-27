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

    const ADVISOR_VRAM = [4, 6, 8, 10, 12, 16, 24, 32, 48, 64, 80];

    const ADVISOR_CAPS = [
        { id: "toolcalls",    icon: "🔧", title: "Tool Calls / MCP",  desc: "Werkzeugaufrufe für Agenten-Workflows",     re: /instruct|\binit\b|\bsft\b|-chat\b|-it\b|rlhf|-dpo\b|assistant/i },
        { id: "thinking",     icon: "🧠", title: "Thinking",          desc: "Extended Reasoning / Chain-of-Thought",     re: null, field: "thinking" },
        { id: "code",         icon: "💻", title: "Code",              desc: "Code generieren und analysieren",           re: /code|coder|starcoder|devstral|codestral/i },
        { id: "multilingual", icon: "🌍", title: "Mehrsprachig",      desc: "Optimiert für Nicht-Englische Texte",       re: /qwen|eurollm|aya|bloom|mistral.nemo|multilingual|deutsch|euro/i },
        { id: "vision",       icon: "👁",  title: "Vision / Multimodal", desc: "Bildverständnis – multimodale Eingaben", re: /vision|pixtral|llava|bakllava|minicpm.v|qwen.*vl|internvl|cogvlm|moondream|paligemma/i },
        { id: "embedding",    icon: "🔢", title: "Embedding / RAG",   desc: "Vektorerzeugung für Suche und RAG",         re: /embed|bge|nomic.embed|all-minilm|instructor|gte-|jina.embed|\be5\b/i },
    ];

    const QUAL_PATTERNS = {
        small:   /^(IQ[12]|Q[23]_)/i,
        quality: /^(Q[5-8]|F16|BF16)/i,
    };

    const TOTAL_COLS = 7 + GPU_LIMITS.length;

    let models = [];

    let state = {
        ctx:      Object.keys(CTX_TOKENS)[0],
        mcp:      false,
        thinking: false,
        quant:    "",
        search:  "",
        hideRed: {},   // { 6: true, 12: false, … }
        sortCol: "name",
        sortDir: 1,
        advisor: { vramGb: null, capabilities: [], archPref: "", qualPref: "" },
    };

    let advisorDraft = { vramGb: null, ctxName: null, capabilities: [], archPref: "", qualPref: "" };

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
            case "ctx":   return model.n_ctx_orig || 0;
            case "total": return v.total;
            default:      return model.name.toLowerCase();
        }
    }

    // ── Render ───────────────────────────────────────────────

    function renderTable() {
        const tbody = document.getElementById("model-tbody");
        const q = state.search.toLowerCase();

        const adv = state.advisor;

        const MCP_RE = /instruct|\binit\b|\bsft\b|-chat\b|-it\b|rlhf|-dpo\b|assistant/i;

        let rows = models.filter(m => {
            const text = (m.name + " " + m.key).toLowerCase();
            if (state.mcp      && !MCP_RE.test(text)) return false;
            if (state.thinking && !m.thinking)         return false;
            if (state.quant && m.quant !== state.quant) return false;
            if (q && !m.name.toLowerCase().includes(q) && !m.arch.toLowerCase().includes(q)) return false;
            if (isHiddenByRed(m)) return false;
            // Advisor filters
            if (adv.vramGb) {
                const total = (m.vram[state.ctx] || { total: 0 }).total;
                if (total > adv.vramGb) return false;
            }
            if (adv.archPref === "dense" && m.moe)  return false;
            if (adv.archPref === "moe"   && !m.moe) return false;
            if (adv.qualPref && m.quant) {
                const pat = QUAL_PATTERNS[adv.qualPref];
                if (pat && !pat.test(m.quant)) return false;
            }
            if (adv.capabilities.length) {
                const text = (m.name + " " + m.key).toLowerCase();
                if (!adv.capabilities.every(function (id) {
                    const cap = ADVISOR_CAPS.find(function (c) { return c.id === id; });
                    if (!cap) return true;
                    return cap.field ? !!m[cap.field] : cap.re.test(text);
                })) return false;
            }
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

            return [
                "<tr>",
                '<td class="col-name" title="' + m.key + '"' + ctxOver + ">" + m.name + "</td>",
                "<td><span class='badge badge-arch'>" + m.arch + "</span>" + moe + "</td>",
                "<td>" + (m.quant || "—") + "</td>",
                '<td class="col-mono col-muted">' + m.size_gb.toFixed(2) + " GB</td>",
                '<td class="col-mono">' + (m.isSSM ? '<span class="badge badge-arch">SSM</span>' : v.kv.toFixed(2) + " GB") + "</td>",
                '<td class="col-mono col-muted">' + fmtCtx(m.n_ctx_orig) + "</td>",
                '<td class="col-mono">' + v.total.toFixed(2) + " GB</td>",
                gpuCells,
                "</tr>",
            ].join("");
        }).join("");
    }

    // ── Filters & sort init ──────────────────────────────────

    function populateFilters() {
        const quants = [...new Set(models.map(m => m.quant).filter(Boolean))].sort();

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

    // ── Advisor ──────────────────────────────────────────────

    function openAdvisor() {
        advisorDraft = {
            vramGb:       state.advisor.vramGb,
            ctxName:      state.ctx,
            capabilities: state.advisor.capabilities.slice(),
            archPref:     state.advisor.archPref,
            qualPref:     state.advisor.qualPref,
        };
        syncAdvisorUI();
        document.getElementById("advisor-overlay").classList.remove("hidden");
    }

    function closeAdvisor() {
        document.getElementById("advisor-overlay").classList.add("hidden");
    }

    function syncAdvisorUI() {
        document.querySelectorAll("#adv-vram .pill").forEach(function (p) {
            p.classList.toggle("selected", Number(p.dataset.gb) === advisorDraft.vramGb);
        });
        document.querySelectorAll("#adv-ctx .pill").forEach(function (p) {
            p.classList.toggle("selected", p.dataset.ctx === advisorDraft.ctxName);
        });
        document.querySelectorAll("#adv-cap .cap-card").forEach(function (c) {
            c.classList.toggle("selected", advisorDraft.capabilities.includes(c.dataset.cap));
        });
        document.querySelectorAll("#adv-arch .tog-btn").forEach(function (b) {
            b.classList.toggle("selected", b.dataset.val === advisorDraft.archPref);
        });
        document.querySelectorAll("#adv-qual .tog-btn").forEach(function (b) {
            b.classList.toggle("selected", b.dataset.val === advisorDraft.qualPref);
        });
    }

    function updateAdvisorBtn() {
        const adv = state.advisor;
        const active = adv.vramGb || adv.capabilities.length || adv.archPref || adv.qualPref;
        const btn = document.getElementById("advisor-btn");
        btn.classList.toggle("adv-active", !!active);
        btn.textContent = active ? "⚡ Berater ✓" : "⚡ Berater";
    }

    function buildAdvisor() {
        // VRAM pills
        const vramEl = document.getElementById("adv-vram");
        ADVISOR_VRAM.forEach(function (gb) {
            const btn = document.createElement("button");
            btn.className = "pill";
            btn.dataset.gb = gb;
            btn.textContent = gb + " GB";
            btn.addEventListener("click", function () {
                advisorDraft.vramGb = advisorDraft.vramGb === gb ? null : gb;
                syncAdvisorUI();
            });
            vramEl.appendChild(btn);
        });

        // Context pills
        const ctxEl = document.getElementById("adv-ctx");
        Object.keys(CTX_TOKENS).forEach(function (name) {
            const btn = document.createElement("button");
            btn.className = "pill";
            btn.dataset.ctx = name;
            btn.textContent = name;
            btn.addEventListener("click", function () {
                advisorDraft.ctxName = name;
                syncAdvisorUI();
            });
            ctxEl.appendChild(btn);
        });

        // Capability cards
        const capEl = document.getElementById("adv-cap");
        ADVISOR_CAPS.forEach(function (cap) {
            const card = document.createElement("div");
            card.className = "cap-card";
            card.dataset.cap = cap.id;
            card.innerHTML =
                '<div class="cap-icon">' + cap.icon + '</div>' +
                '<div class="cap-title">' + cap.title + '</div>' +
                '<div class="cap-desc">'  + cap.desc  + '</div>';
            card.addEventListener("click", function () {
                const idx = advisorDraft.capabilities.indexOf(cap.id);
                if (idx === -1) advisorDraft.capabilities.push(cap.id);
                else            advisorDraft.capabilities.splice(idx, 1);
                syncAdvisorUI();
            });
            capEl.appendChild(card);
        });

        // Arch toggle
        const archEl = document.getElementById("adv-arch");
        [{ label: "Alle", val: "" }, { label: "Dense", val: "dense" }, { label: "MoE", val: "moe" }]
            .forEach(function (opt) {
                const btn = document.createElement("button");
                btn.className = "tog-btn";
                btn.dataset.val = opt.val;
                btn.textContent = opt.label;
                btn.addEventListener("click", function () {
                    advisorDraft.archPref = opt.val;
                    syncAdvisorUI();
                });
                archEl.appendChild(btn);
            });

        // Quality toggle
        const qualEl = document.getElementById("adv-qual");
        [{ label: "Kleinstmöglich", val: "small" }, { label: "Ausgewogen", val: "" }, { label: "Beste Qualität", val: "quality" }]
            .forEach(function (opt) {
                const btn = document.createElement("button");
                btn.className = "tog-btn";
                btn.dataset.val = opt.val;
                btn.textContent = opt.label;
                btn.addEventListener("click", function () {
                    advisorDraft.qualPref = opt.val;
                    syncAdvisorUI();
                });
                qualEl.appendChild(btn);
            });

        // Buttons
        document.getElementById("advisor-btn").addEventListener("click", openAdvisor);
        document.getElementById("advisor-close").addEventListener("click", closeAdvisor);
        document.getElementById("advisor-cancel").addEventListener("click", closeAdvisor);
        document.getElementById("advisor-overlay").addEventListener("click", function (e) {
            if (e.target === e.currentTarget) closeAdvisor();
        });
        document.addEventListener("keydown", function (e) {
            if (e.key === "Escape" && !document.getElementById("advisor-overlay").classList.contains("hidden")) {
                closeAdvisor();
            }
        });

        document.getElementById("advisor-apply").addEventListener("click", function () {
            state.advisor.vramGb       = advisorDraft.vramGb;
            state.advisor.capabilities = advisorDraft.capabilities.slice();
            state.advisor.archPref     = advisorDraft.archPref;
            state.advisor.qualPref     = advisorDraft.qualPref;
            if (advisorDraft.ctxName) {
                state.ctx = advisorDraft.ctxName;
                document.getElementById("ctx-select").value = advisorDraft.ctxName;
            }
            updateAdvisorBtn();
            closeAdvisor();
            renderTable();
        });

        document.getElementById("advisor-reset").addEventListener("click", function () {
            state.advisor = { vramGb: null, capabilities: [], archPref: "", qualPref: "" };
            updateAdvisorBtn();
            closeAdvisor();
            renderTable();
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
                document.getElementById("filter-mcp").addEventListener("change", function (e) {
                    state.mcp = e.target.checked; renderTable();
                });
                document.getElementById("filter-thinking").addEventListener("change", function (e) {
                    state.thinking = e.target.checked; renderTable();
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

                buildAdvisor();
                renderTable();
            })
            .catch(function (err) {
                document.getElementById("model-tbody").innerHTML =
                    '<tr><td colspan="' + TOTAL_COLS + '" class="no-results">Fehler beim Laden: ' + err.message + '</td></tr>';
            });
    });
}());
