// ── Graph ──

function buildLegend() {
  document.getElementById("glegend").innerHTML =
    Object.entries(CATS).map(([k,v]) =>
      `<div class="gli"><div class="gldot" style="background:${v.color}"></div>${k}</div>`
    ).join("") +
    `<div class="gli"><div class="glsq" style="background:transparent;border:2px dashed var(--c-data)"></div>Dataset</div>`;
}

function renderGraph() {
  const showComp    = document.getElementById("tog-comp").checked;
  const showData    = document.getElementById("tog-data").checked;
  const showSameCat = document.getElementById("tog-cat").checked;

  let methods = METHODS.filter(m => !m.is_placeholder && isComp(m));
  if (gStageFilter !== "All") {
    methods = methods.filter(m =>
      (m.pipeline_category || "").toLowerCase().includes(gStageFilter.toLowerCase())
    );
  }

  const maxCit    = Math.max(1, ...methods.map(m => parseInt(m.citations) || 0));
  const methodIds = new Set(methods.map(m => m.id));
  const elements  = [];

  // Group nodes by category
  const catNodes = {};
  methods.forEach(m => {
    const key = m.pipeline_category || "Other";
    if (!catNodes[key]) catNodes[key] = [];
    catNodes[key].push(m);
  });

  // Place nodes: sunflower spiral around each category centroid.
  // methodPos is kept around so the dataset block below can anchor dataset
  // nodes near the methods that actually use them, instead of a fixed row.
  const methodPos = {};
  methods.forEach(m => {
    const info  = catInfo(m.pipeline_category);
    const cit   = parseInt(m.citations) || 0;
    const size  = 14 + Math.round((cit / maxCit) * 24);
    const key   = m.pipeline_category || "Other";
    const list  = catNodes[key];
    const idx   = list.indexOf(m);
    const angle  = idx * 2.399;
    const radius = list.length === 1 ? 0 : 25 + Math.sqrt(idx) * 22;
    const px = info.x + radius * Math.cos(angle);
    const py = info.y + radius * Math.sin(angle);
    methodPos[m.id] = { x: px, y: py };
    elements.push({
      data: { id: m.id, label: m.name || m.id, color: info.color, size, type: "method", _m: JSON.stringify(m) },
      position: { x: px, y: py }
    });
  });

  // Fully connected within-category edges
  if (showSameCat) {
    Object.values(catNodes).forEach(list => {
      for (let i = 0; i < list.length; i++) {
        for (let j = i + 1; j < list.length; j++) {
          elements.push({
            data: { id: `cat_${list[i].id}_${list[j].id}`, source: list[i].id, target: list[j].id, type: "cat-edge" }
          });
        }
      }
    });
  }

  // Cross-category edges: connect first node of each category to first node of adjacent categories
  // Pipeline adjacency: Preprocessing→Segmentation→Phenotyping→Niche→SVG/CCC
  const catAdj = [
    ["Preprocessing", "Cell segmentation - Imaging based"],
    ["Preprocessing", "Cell segmentation - Transcript based"],
    ["Cell segmentation - Imaging based", "Phenotyping"],
    ["Cell segmentation - Transcript based", "Phenotyping"],
    ["Cell segmentation - Imaging based", "Cell segmentation - Transcript based"],
    ["Phenotyping", "Niche/Neighborhood/domain analysis"],
    ["Phenotyping", "Cell type Deconvolution"],
    ["Niche/Neighborhood/domain analysis", "Cell-Cell-Communication"],
    ["Niche/Neighborhood/domain analysis", "Spatial Variable Genes"],
    ["Niche/Neighborhood/domain analysis", "Label separation/pattern extraction - ML"],
  ];
  if (showSameCat) {
    catAdj.forEach(([a, b]) => {
      const listA = catNodes[a], listB = catNodes[b];
      if (listA && listB && listA.length && listB.length) {
        elements.push({
          data: { id: `bridge_${a}_${b}`, source: listA[0].id, target: listB[0].id, type: "bridge-edge" }
        });
      }
    });
  }

  // Comparison edges
  if (showComp) {
    const added = new Set();
    methods.forEach(m => {
      (m.comparison_ids || []).forEach(cid => {
        cid = cid.trim();
        if (cid && methodIds.has(cid) && cid !== m.id) {
          const eid = `c_${[m.id, cid].sort().join("_")}`;
          if (!added.has(eid)) { added.add(eid); elements.push({ data: { id: eid, source: m.id, target: cid, type: "comp" } }); }
        }
      });
    });
  }

  // Dataset nodes + edges.
  // Previously: dataset nodes were placed in one long row (x:100+i*110,
  // y:900), completely unrelated to which methods used them. Method nodes
  // live in a tight x:180-1100, y:130-580 cluster, so with more than a
  // handful of datasets that row stretched the bounding box enormously -
  // cy.fit() then zoomed out to fit it, shrinking the whole method graph to
  // near-invisible dots. This is the "toggling datasets doesn't work" bug.
  // Fix: anchor each dataset node at the centroid of the method node(s)
  // that actually link to it, with a small radial offset so it doesn't sit
  // exactly on top of them.
  if (showData) {
    const usedIds = new Set();
    methods.forEach(m => (m.data_ids || []).forEach(d => { if (d.trim()) usedIds.add(d.trim()); }));
    const dsMap = {};
    DATASETS.forEach(d => { if (usedIds.has(d.id)) dsMap[d.id] = d; });

    const dsLinks = {};
    methods.forEach(m => (m.data_ids || []).forEach(did => {
      did = did.trim();
      if (did && dsMap[did]) (dsLinks[did] = dsLinks[did] || []).push(m.id);
    }));

    Object.entries(dsMap).forEach(([id, d], i) => {
      const linkedIds = dsLinks[id] || [];
      const pts = linkedIds.map(mid => methodPos[mid]).filter(Boolean);
      const cx  = pts.length ? pts.reduce((s, p) => s + p.x, 0) / pts.length : 640;
      const cyy = pts.length ? pts.reduce((s, p) => s + p.y, 0) / pts.length : 350;
      const angle = i * 2.399;
      const r = 30;
      elements.push({
        data: { id: d.id, label: d.internal_name || d.id, type: "dataset", _d: JSON.stringify(d) },
        position: { x: cx + r * Math.cos(angle), y: cyy + r * Math.sin(angle) }
      });
    });

    methods.forEach(m => {
      (m.data_ids || []).forEach(did => {
        did = did.trim();
        if (did && dsMap[did]) elements.push({ data: { id: `de_${m.id}_${did}`, source: m.id, target: did, type: "data-edge" } });
      });
    });
  }

  if (cy) cy.destroy();
  cy = cytoscape({
    container: document.getElementById("graph-container"),
    elements,
    style: [
      { selector: "node[type='method']", style: {
        "background-color":   "data(color)",
        "width": "data(size)", "height": "data(size)",
        "label": "data(label)", "color": "#e6edf3",
        "font-size": "9px", "font-family": "Inter,sans-serif",
        "text-valign": "bottom", "text-margin-y": 4,
        "text-outline-width": 1.5, "text-outline-color": "#0d1117",
        "border-width": 1.5, "border-color": "#21262d",
      }},
      { selector: "node[type='dataset']", style: {
        // Dashed-outline "entity" look (dark fill, dashed green border)
        // instead of a solid green square, so datasets read visually as a
        // different kind of thing from method nodes rather than just
        // another colored dot.
        "background-color": "#0d1117", "shape": "round-rectangle",
        "border-width": 2, "border-style": "dashed", "border-color": "#39d353",
        "width": 14, "height": 14,
        "label": "data(label)", "color": "#e6edf3",
        "font-size": "8px", "font-family": "JetBrains Mono,monospace",
        "text-valign": "bottom", "text-margin-y": 3,
        "text-outline-width": 1, "text-outline-color": "#0d1117",
      }},
      { selector: "edge[type='cat-edge']", style: {
        "line-color": "data(color)", "width": 0.4, "opacity": 0.12,
        "curve-style": "bezier",
      }},
      { selector: "edge[type='bridge-edge']", style: {
        "line-color": "#555", "width": 0.6, "opacity": 0.2,
        "curve-style": "bezier", "line-style": "dashed",
      }},
      { selector: "edge[type='comp']", style: {
        "line-color": "#388bfd", "width": 1.5, "opacity": 0.6,
        "curve-style": "bezier",
        "target-arrow-shape": "triangle", "target-arrow-color": "#388bfd", "arrow-scale": 0.7,
      }},
      { selector: "edge[type='data-edge']", style: {
        "line-color": "#39d353", "line-style": "dashed",
        "width": 1, "opacity": 0.35, "curve-style": "bezier",
      }},
      { selector: "node:selected", style: { "border-width": 3, "border-color": "#fff" }},
    ],
    layout: { name: "preset", animate: false },
    userZoomingEnabled: true,
    userPanningEnabled: true,
  });

  cy.fit(cy.nodes(), 40);

  cy.on("tap", "node", evt => {
    const d = evt.target.data();
    try {
      if (d.type === "method"  && d._m) showNodeDetail(JSON.parse(d._m));
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
    c += m.comparison_ids.map(id => { const o = METHODS.find(x => x.id === id); return `<div>· ${o ? (o.name || id) : id}</div>`; }).join("");
  }
  if ((m.data_ids || []).length) {
    c += `<strong style="margin-top:0.4rem;display:block;">Datasets (${m.data_ids.length})</strong>`;
    c += m.data_ids.map(id => { const ds = DATASETS.find(x => x.id === id); return `<div>· ${ds ? (ds.internal_name || id) : id}</div>`; }).join("");
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
