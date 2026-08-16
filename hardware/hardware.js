/*
 * Gemeinsame Render-/Sortier-Logik für alle hardware/<Gerät>/index.html Seiten.
 *
 * Erwartet vor dem Einbinden dieser Datei ein globales `MODELS`-Array mit Zeilen
 * in folgendem festen Schema:
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

  const COL_INDEX = { name: 0, params: 1, type: 2, quant: 3, vram: 4, speed: 6, ctx: 8, status: 10, score: 12 };

  const tbody = document.getElementById("model-tbody");
  const rowCount = document.getElementById("row-count");
  const searchInput = document.getElementById("search-input");

  let sortCol = null;
  let sortDir = 1;
  let search = "";

  function statusClass(key) {
    return "status-" + key;
  }

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
      const [name, params, type, quant, vram, vramPct, speed, ctxLabel, ctxNum, statusKey, statusText, grade, score] = r;
      const types = type.split(", ").map(t => '<span class="type-tag">' + t + '</span>').join("");
      const barColor = vramPct >= 90 ? "var(--red)" : vramPct >= 60 ? "var(--yellow)" : "var(--green)";
      return [
        "<tr>",
        '<td class="col-name">' + name + "</td>",
        '<td class="col-mono col-muted">' + params + " B</td>",
        "<td>" + types + "</td>",
        '<td class="col-mono">' + quant + "</td>",
        '<td><div class="vram-bar-wrap"><span class="col-mono">' + vram.toFixed(1) + ' GB</span><div class="vram-bar"><div class="vram-bar-fill" style="width:' + Math.min(vramPct, 100) + '%;background:' + barColor + '"></div></div><span class="col-mono col-muted">' + vramPct + '%</span></div></td>',
        '<td class="col-mono">' + speed.toFixed(1) + " t/s</td>",
        '<td class="col-mono col-muted">' + ctxLabel + "</td>",
        '<td><span class="status-badge ' + statusClass(statusKey) + '">' + statusText + "</span></td>",
        '<td><span class="grade-badge"><span class="grade-letter grade-' + grade + '">' + grade + '</span><span class="grade-score">' + score + "</span></span></td>",
        "</tr>",
      ].join("");
    }).join("");
  }

  document.querySelectorAll("th.sortable").forEach(th => {
    th.addEventListener("click", () => {
      const col = COL_INDEX[th.dataset.col];
      if (sortCol === col) {
        sortDir *= -1;
      } else {
        sortCol = col;
        sortDir = 1;
      }
      document.querySelectorAll("th[data-sorted]").forEach(t => t.removeAttribute("data-sorted"));
      th.setAttribute("data-sorted", sortDir === 1 ? "asc" : "desc");
      render();
    });
  });

  searchInput.addEventListener("input", () => {
    search = searchInput.value;
    render();
  });

  render();
})();
