// ── Graph ──

function buildLegend() {
  document.getElementById("glegend").innerHTML =
    Object.entries(CATS).map(([k,v]) =>
      `<div class="gli"><div class="gldot" style="background:${v.color}"></div>${k}</div>`
    ).join("") +
    `<div class="gli"><div class="glsq" style="background:var(--c-data)"></div>Dataset</div>`;
}

function renderGraph() {
  const showComp = document.getElementById("tog-comp").checked;
  const showData = document.getElementById("tog-data").checked;

  let methods = METHODS.filter(m => !m.is_placeholder && isComp(m));
  if (gStageFilter !== "All") {
    methods = methods.filter(m =>
      (m.pipeline_category || "").toLowerCase().includes(gStageFilter.toLowerCase())
    );
  }

  const maxCit = Math.max(1, ...methods.map(m => parseInt(m.citations) || 0));
  const methodIds = new Set(methods.map(m => m.id));
  const elements = [];

  // Track how many nodes already placed per category for jitter spacing
  const catCount = {};

  methods.forEach(m => {
    const info = catInfo(m.pipeline_category);
    const cit  = parseInt(m.citations) || 0;
    const size = 16 + Math.round((cit / maxCit) * 26);

    const key = m.pipeline_category || "Other";
    catCount[key] = (catCount[key] || 0) + 1;
    const n = catCount[key];

    // Spiral jitter around centroid so nodes spread out evenly
    const angle  = n * 2.399; // golden angle
    const radius = 20 + n * 18;
    const x = info.x + radius * Math.cos(angle);
    const y = info.y + radius * Math.sin(angle);

    elements.push({
      data: { id: m.id, label: m.name || m.id, color: info.color, size, type: "method", _m: JSON.stringify(m) },
      position: { x, y }
    });
  });

  // Comparison edges
  if (showComp) {
    methods.forEach(m => {
      (m.comparison_ids || []).forEach(cid => {
        cid = cid.trim();
        if (cid && methodIds.has(cid) && cid !== m.id) {
          const eid = `c_${m.id}_${cid}`;
          if (!elements.find(e => e.data && e.data.id === eid)) {
            elements.push({ data: { id: eid, source: m.id, target: cid, type: "comp" } });
          }
        }
      });
    });
  }

  // Dataset nodes + edges
  if (showData) {
    const usedIds = new Set();
    methods.forEach(m => (m.data_ids || []).forEach(d => { if (d.trim()) usedIds.add(d.trim()); }));
    const dsMap = {};
    DATASETS.forEach(d => { if (usedIds.has(d.id)) dsMap[d.id] = d; });

    Object.values(dsMap).forEach((d, i) => {
      elements.push({
        data: { id: d.id, label: d.internal_name || d.id, type: "dataset", _d: JSON.stringify(d) },
        position: { x: 80 + i * 110, y: 720 }
      });
    });

    methods.forEach(m => {
      (m.data_ids || []).forEach(did => {
        did = did.trim();
        if (did && dsMap[did]) {
          elements.push({ data: { id: `de_${m.id}_${did}`, source: m.id, target: did, type: "data-edge" } });
        }
      });
    });
  }

  if (cy) cy.destroy();

  cy = cytoscape({
    container: document.getElementById("graph-container"),
    elements,
    style: [
      { selector: "node[type='method']", style: {
        "background-color":    "data(color)",
        "width":               "data(size)", "height": "data(size)",
        "label":               "data(label)", "color": "#e6edf3",
        "font-size":           "9px", "font-family": "Inter,sans-serif",
        "text-valign":         "bottom", "text-margin-y": 4,
        "text-outline-width":  1.5, "text-outline-color": "#0d1117",
        "border-width":        1.5, "border-color": "#21262d",
      }},
      { selector: "node[type='dataset']", style: {
        "background-color": "#39d353", "shape": "rectangle",
        "width": 14, "height": 14,
        "label": "data(label)", "color": "#e6edf3",
        "font-size": "8px", "font-family": "JetBrains Mono,monospace",
        "text-valign": "bottom", "text-margin-y": 3,
        "text-outline-width": 1, "text-outline-color": "#0d1117",
      }},
      { selector: "edge[type='comp']", style: {
        "line-color": "#388bfd", "width": 1.5, "opacity": 0.55,
        "curve-style": "bezier",
        "target-arrow-shape": "triangle", "target-arrow-color": "#388bfd", "arrow-scale": 0.7,
      }},
      { selector: "edge[type='data-edge']", style: {
        "line-color": "#39d353", "line-style": "dashed",
        "width": 1, "opacity": 0.35, "curve-style": "bezier",
      }},
      { selector: "node:selected", style: { "border-width": 3, "border-color": "#fff" }},
    ],
    // Pure preset — no physics, no force layout, positions are final
    layout: { name: "preset", animate: false }
  });

  cy.on("tap", "node", evt => {
    const d = evt.target.data();
    try {
      if (d.type === "method" && d._m) showNodeDetail(JSON.parse(d._m));
      else if (d.type === "dataset" && d._d) showDsDetail(JSON.parse(d._d));
    } catch(e) { console.error(e); }
  });
  cy.on("tap", evt => { if (evt.target === cy) closeDetail(); });
}

function showNodeDetail(m) {
  const p = document.getElementById("dpanel");
  document.getElementById("d-id").textContent   = m.id;
  document.getElementById("d-name").textContent = m.name || m.title || m.id;
  document.getElementById("d-sub").textContent  = [m.first_author, m.journal, m.year].filter(Boolean).join(" · ");
  document.getElementById("d-abs").textContent  = m.abstract || "";
  document.getElementById("d-doi").innerHTML    = m.doi ? `<a href="${m.doi}" target="_blank">${m.doi}</a>` : "";
  let c = "";
  if ((m.comparison_ids || []).length) {
    c += `<strong>Compared against (${m.comparison_ids.length})</strong>`;
    c += m.comparison_ids.map(id => {
      const o = METHODS.find(x => x.id === id);
      return `<div>· ${o ? (o.name || id) : id}</div>`;
    }).join("");
  }
  if ((m.data_ids || []).length) {
    c += `<strong style="margin-top:0.4rem;display:block;">Datasets (${m.data_ids.length})</strong>`;
    c += m.data_ids.map(id => {
      const ds = DATASETS.find(x => x.id === id);
      return `<div>· ${ds ? (ds.internal_name || id) : id}</div>`;
    }).join("");
  }
  document.getElementById("d-conns").innerHTML = c;
  p.classList.add("on");
}

function showDsDetail(d) {
  document.getElementById("d-id").textContent   = d.id;
  document.getElementById("d-name").textContent = d.internal_name || d.id;
  document.getElementById("d-sub").textContent  = [d.spatial_data_method, d.tissue, d.disease, d.organism].filter(Boolean).join(" · ");
  document.getElementById("d-abs").textContent  = d.notes || "";
  document.getElementById("d-doi").innerHTML    = d.access_link ? `<a href="${d.access_link}" target="_blank">↗ Access data</a>` : "";
  document.getElementById("d-conns").innerHTML  = "";
  document.getElementById("dpanel").classList.add("on");
}

function closeDetail() { document.getElementById("dpanel").classList.remove("on"); }

function openInGraph(id) {
  showSection("graph");
  setTimeout(() => {
    if (!cy) return;
    const node = cy.getElementById(id);
    if (node.length) {
      cy.animate({ fit: { eles: node, padding: 100 }, duration: 400 });
      const m = METHODS.find(x => x.id === id);
      if (m) showNodeDetail(m);
    }
  }, 500);
}
