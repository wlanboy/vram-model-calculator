// Gemeinsame Helfer für filter.js und hardware/hardware.js.
(function (global) {
    "use strict";

    // Modellnamen, Architekturen und Typen stammen aus GGUF-Metadaten oder
    // Dateinamen Dritter und landen per innerHTML in der Seite. Darum jeden
    // solchen Wert vor dem Einsetzen in Markup escapen.
    const HTML_ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };

    function esc(value) {
        return String(value).replace(/[&<>"']/g, c => HTML_ESCAPES[c]);
    }

    // Sortierbare Spaltenköpfe: Klick, Enter und Leertaste lösen onSort(th)
    // aus; aria-sort und data-sorted zeigen die aktuelle Richtung.
    function initSortableHeaders(onSort) {
        document.querySelectorAll("th.sortable").forEach(function (th) {
            th.tabIndex = 0;
            th.setAttribute("aria-sort", "none");
            th.addEventListener("click", function () { onSort(th); });
            th.addEventListener("keydown", function (e) {
                if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    onSort(th);
                }
            });
        });
    }

    function markSorted(activeTh, dir) {
        document.querySelectorAll("th.sortable").forEach(function (th) {
            if (th === activeTh) {
                th.setAttribute("data-sorted", dir === 1 ? "asc" : "desc");
                th.setAttribute("aria-sort", dir === 1 ? "ascending" : "descending");
            } else {
                th.removeAttribute("data-sorted");
                th.setAttribute("aria-sort", "none");
            }
        });
    }

    global.VramCommon = Object.freeze({ esc, initSortableHeaders, markSorted });
}(window));
