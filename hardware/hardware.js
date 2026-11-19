/*
 * Gemeinsame Render-/Sortier-Logik für alle hardware/<Gerät>/index.html Seiten.
 *
 * Erwartet vor dem Einbinden dieser Datei ../../common.js und ein globales
 * `MODELS`-Array. Das Array erzeugt `uv run main.py hardware` aus
 * hardware/<Gerät>/device.json und models_cache.json, Zeilen im festen Schema:
 *   [name, paramsB, type, quant, vramGb, vramPct, speedTs, ctxLabel, ctxNum,
 *    statusKey, statusText, grade, score]
 *
 * Erwartet im HTML:
 *   <input id="search-input">
 *   <div id="row-count"></div>
 *   <table><thead><tr>
 *     <th class="sortable" data-col="name">…</th>
 *     <th class="sortable" data-col="params">…</th>
 *     <th class="sortable" data-col="type">…</th>
 *     <th class="sortable" data-col="quant">…</th>
 *     <th class="sortable" data-col="vram">…</th>
 *     <th class="sortable" data-col="speed">…</th>
 *     <th class="sortable" data-col="ctx">…</th>
 *     <th class="sortable" data-col="status">…</th>
 *     <th class="sortable" data-col="score">…</th>
 *   </tr></thead><tbody id="model-tbody"></tbody></table>
 */
(function () {
  "use strict";

  const { esc, initSortableHeaders, markSorted } = window.VramCommon;

  const COL_INDEX = { name: 0, params: 1, type: 2, quant: 3, vram: 4, speed: 6, ctx: 8, status: 10, score: 12 };

  const tbody = document.getElementById("model-tbody");
  const rowCount = document.getElementById("row-count");
  const searchInput = document.getElementById("search-input");

  let sortCol = null;
  let sortDir = 1;
  let search = "";

  function render() {
    const q = search.toLowerCase();
    let rows = MODELS.filter(r =>
      !q || r[0].toLowerCase().includes(q) || r[2].toLowerCase().includes(q) || r[3].toLowerCase().includes(q)
    );

    if (sortCol !== null) {
      rows = rows.slice().sort((a, b) => {
        let va = a[sortCol], vb = b[sortCol];
        if (typeof va === "string") va = va.toLowerCase();
        if (typeof vb === "string") vb = vb.toLowerCase();
        if (va < vb) return -sortDir;
        if (va > vb) return sortDir;
        return 0;
      });
    }

    rowCount.textContent = rows.length + " von " + MODELS.length + " Modellen";

    tbody.innerHTML = rows.map(r => {
      const [name, params, type, quant, vram, vramPct, speed, ctxLabel, , statusKey, statusText, grade, score] = r;
      const types = type.split(", ").map(t => '<span class="type-tag">' + esc(t) + '</span>').join("");
      const barColor = vramPct >= 90 ? "var(--red)" : vramPct >= 60 ? "var(--yellow)" : "var(--green)";
      return [
        "<tr>",
        '<td class="col-name">' + esc(name) + "</td>",
        '<td class="col-mono col-muted">' + esc(params) + " B</td>",
        "<td>" + types + "</td>",
        '<td class="col-mono">' + esc(quant) + "</td>",
        '<td><div class="vram-bar-wrap"><span class="col-mono">' + Number(vram).toFixed(1) + ' GB</span><div class="vram-bar"><div class="vram-bar-fill" style="width:' + Math.min(Number(vramPct), 100) + '%;background:' + barColor + '"></div></div><span class="col-mono col-muted">' + esc(vramPct) + '%</span></div></td>',
        '<td class="col-mono">' + Number(speed).toFixed(1) + " t/s</td>",
        '<td class="col-mono col-muted">' + esc(ctxLabel) + "</td>",
        '<td><span class="status-badge status-' + esc(statusKey) + '">' + esc(statusText) + "</span></td>",
        '<td><span class="grade-badge"><span class="grade-letter grade-' + esc(grade) + '">' + esc(grade) + '</span><span class="grade-score">' + esc(score) + "</span></span></td>",
        "</tr>",
      ].join("");
    }).join("");
  }

  initSortableHeaders(th => {
    const col = COL_INDEX[th.dataset.col];
    if (sortCol === col) {
      sortDir *= -1;
    } else {
      sortCol = col;
      sortDir = 1;
    }
    markSorted(th, sortDir);
    render();
  });

  searchInput.addEventListener("input", () => {
    search = searchInput.value;
    render();
  });

  render();
})();
