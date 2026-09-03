// ── Data filters ──
// Defaults to spatial proteomics: the structured dataset registry is
// currently SP-only in scope (per CLAUDE.md) - ST rows are retained in the
// data because they were already curated, not because ST is in scope, and
// they're overwhelmingly missing disease/marker annotation (13/69 ST vs
// 62/72 SP have a disease value), which made the registry look broken when
// both were mixed under "All". Toggle is kept so ST/All are still reachable.
let dFilter="proteomics";
function initDataFilters() {
  const mods=[{k:"All",l:"All"},{k:"proteomics",l:"Spatial Proteomics"},{k:"transcriptomics",l:"Spatial Transcriptomics"}];
  document.getElementById("data-filters").innerHTML=mods.map(m=>
    `<button class="pill ${m.k===dFilter?"active":""}" onclick="setDF('${m.k}',this)">${m.l}</button>`
  ).join("");
}
function setDF(v,btn) {
  dFilter=v;
  document.querySelectorAll("#data-filters .pill").forEach(p=>p.classList.remove("active"));
  btn.classList.add("active"); renderDatasets();
}

// ── Disease/tissue/marker filters (canonicalized - see tissue_disease_maps.py
// and 03_export_json.py's markers_list) ──
let diseaseFilter="All", tissueFilter="All", markerFilter="All";

function initDiseaseFilters() {
  const diseases=["All",...Object.keys(STATS.disease_clean_counts||{}).sort()];
  document.getElementById("data-disease-filters").innerHTML=diseases.map(d=>
    `<button class="pill ${d==="All"?"active":""}" onclick="setDiseaseFilter('${d}',this)">${d}</button>`
  ).join("");
}
function setDiseaseFilter(v,btn) {
  diseaseFilter=v;
  document.querySelectorAll("#data-disease-filters .pill").forEach(p=>p.classList.remove("active"));
  btn.classList.add("active"); renderDatasets();
}

// Both dropdowns are sorted by real usage frequency (most common tissue/
// marker first), not alphabetically - matches how marker_counts/
// tissue_counts were built specifically for this.
function initTissueFilter() {
  const entries=Object.entries(STATS.tissue_counts||{}).sort((a,b)=>b[1]-a[1]);
  const sel=document.getElementById("data-tissue-filter");
  sel.innerHTML=`<option value="All">All tissues</option>`+
    entries.map(([t,n])=>`<option value="${t}">${t} (${n})</option>`).join("");
}
function setTissueFilter(v) { tissueFilter=v; renderDatasets(); }

function initMarkerFilter() {
  const entries=Object.entries(STATS.marker_counts||{}).sort((a,b)=>b[1]-a[1]);
  const sel=document.getElementById("data-marker-filter");
  sel.innerHTML=`<option value="All">All markers</option>`+
    entries.map(([m,n])=>`<option value="${m}">${m} (${n})</option>`).join("");
}
function setMarkerFilter(v) { markerFilter=v; renderDatasets(); }


// ── Graph filters ──
function initGraphFilters() {
  const stages=["All",...Object.keys(CATS)];
  document.getElementById("graph-filters").innerHTML=stages.map(s=>
    `<button class="pill ${s==="All"?"active":""}" onclick="setGF('${s}',this)">${s}</button>`
  ).join("");
}
function setGF(v,btn) {
  gStageFilter=v;
  document.querySelectorAll("#graph-filters .pill").forEach(p=>p.classList.remove("active"));
  btn.classList.add("active"); renderGraph();
}

// ── Source-type filter (peer-reviewed / bioRxiv / arXiv / preprint) ──
// Combines Marta's manual "arxiv/bioarxiv/peer reviewed" curation with the
// auto-fetched Crossref publication_type (see compute_source_type in
// 03_export_json.py) - "Unknown" covers rows neither source could place.
function initSourceFilters() {
  const opts=[{k:"All",l:"All"},{k:"peer-reviewed",l:"Peer-reviewed"},
              {k:"bioRxiv",l:"bioRxiv"},{k:"arXiv",l:"arXiv"},
              {k:"preprint",l:"Preprint (other)"},{k:"",l:"Unknown"}];
  document.getElementById("source-filters").innerHTML=opts.map(o=>
    `<button class="pill ${o.k==="All"?"active":""}" onclick="setSF('${o.k}',this)">${o.l}</button>`
  ).join("");
}
function setSF(v,btn) {
  gSourceFilter=v;
  document.querySelectorAll("#source-filters .pill").forEach(p=>p.classList.remove("active"));
  btn.classList.add("active"); renderGraph();
}


// ── Sort ──
function sortT(which,col) {
  if(which==="m") { if(mSort.col===col) mSort.dir*=-1; else{mSort.col=col;mSort.dir=1;} renderMethods(); }
  else            { if(dSort.col===col) dSort.dir*=-1; else{dSort.col=col;dSort.dir=1;} renderDatasets(); }
}


// ── Methods ──
// Was a substring check on the free-text `category` field ("computational");
// switched to `paper_type`, which 01_parse_excel.py already sets directly
// from which sheet a row came from (method_pub -> "method") rather than
// deriving it from category text - more robust, and category values no
// longer reliably contain "computational" after the 2026-09-01 cleanup.
function isComp(m) { return m.paper_type === "method"; }

function renderMethods() {
  const q=(document.getElementById("methods-search").value||"").toLowerCase();
  let items=METHODS.filter(m=>!m.is_placeholder&&isComp(m));
  if(q) items=items.filter(m=>
    (m.name||"").toLowerCase().includes(q)||
    (m.id||"").toLowerCase().includes(q)||
    (m.title||"").toLowerCase().includes(q)||
    (m.first_author||"").toLowerCase().includes(q)||
    (m.journal||"").toLowerCase().includes(q)
  );
  const col=mSort.col, dir=mSort.dir;
  items.sort((a,b)=>{
    const av=String(a[col]||""), bv=String(b[col]||"");
    if(col==="citations"||col==="year") return ((parseInt(av)||0)-(parseInt(bv)||0))*dir;
    return av.localeCompare(bv)*dir;
  });
  const tb=document.getElementById("methods-tbody");
  if(!items.length){tb.innerHTML=`<tr><td colspan="9" class="empty-state">No methods found.</td></tr>`;return;}
  tb.innerHTML=items.map(m=>{
    const info=catInfo(m.pipeline_category), cat=(m.pipeline_category||"—").split(";")[0].trim();
    const comps=(m.comparison_ids||[]).length, datas=(m.data_ids||[]).length;
    return `<tr onclick="openInGraph('${m.id}')">
      <td class="tid">${m.id}</td>
      <td class="tname">${m.name||m.id}</td>
      <td><span class="cpill" style="border-left:3px solid ${info.color};padding-left:0.4rem;">${cat}</span></td>
      <td>${m.year||"—"}</td>
      <td>${m.journal?m.journal.slice(0,28)+(m.journal.length>28?"…":""):"—"}</td>
      <td>${m.citations||"—"}</td>
      <td>${comps?`<span class="cbadge comp">${comps}</span>`:"—"}</td>
      <td>${datas?`<span class="cbadge data">${datas}</span>`:"—"}</td>
      <td><span class="sdot ${m.review_status==="manual"?"manual":""}" title="${m.review_status||"stub"}"></span></td>
    </tr>`;
  }).join("");
}


// ── Datasets ──
function renderDatasets() {
  const q=(document.getElementById("data-search").value||"").toLowerCase();
  let items=DATASETS;
  if(dFilter!=="All") items=items.filter(d=>(d.spatial_data_category||"").toLowerCase().includes(dFilter));
  if(diseaseFilter!=="All") items=items.filter(d=>(d.disease_list||[]).includes(diseaseFilter));
  if(tissueFilter!=="All") items=items.filter(d=>(d.tissue_list||[]).includes(tissueFilter));
  if(markerFilter!=="All") items=items.filter(d=>(d.markers_list||[]).includes(markerFilter));
  if(q) items=items.filter(d=>
    (d.id||"").toLowerCase().includes(q)||
    (d.internal_name||"").toLowerCase().includes(q)||
    (d.tissue||"").toLowerCase().includes(q)||
    (d.disease||"").toLowerCase().includes(q)||
    (d.organism||"").toLowerCase().includes(q)||
    (d.spatial_data_method||"").toLowerCase().includes(q)||
    // canonicalized fields - lets search reach through to normalized
    // values/specifics even when the raw scalar text above doesn't
    // literally contain the query (e.g. "TNBC" against a raw disease cell
    // worded differently but captured in disease_specifics_list)
    (d.tissue_list||[]).some(t=>t.toLowerCase().includes(q))||
    (d.disease_list||[]).some(x=>x.toLowerCase().includes(q))||
    (d.disease_specifics_list||[]).some(x=>x.toLowerCase().includes(q))||
    (d.markers_list||[]).some(x=>x.toLowerCase().includes(q))
  );
  const col=dSort.col, dir=dSort.dir;
  items.sort((a,b)=>String(a[col]||"").localeCompare(String(b[col]||""))*dir);
  const tb=document.getElementById("datasets-tbody");
  if(!items.length){tb.innerHTML=`<tr><td colspan="10" class="empty-state">No datasets found.</td></tr>`;return;}
  tb.innerHTML=items.map(d=>{
    const isSP=(d.spatial_data_category||"").toLowerCase().includes("proteomics");
    const feat=d.n_markers||d.n_genes||"—";
    return `<tr>
      <td class="tid">${d.id}</td>
      <td class="tname">${d.internal_name||d.id}</td>
      <td>${isSP?`<span class="cbadge comp">SP</span>`:`<span class="cbadge data">ST</span>`}</td>
      <td>${d.spatial_data_method||"—"}</td>
      <td>${d.organism||"—"}</td>
      <td>${d.tissue||"—"}</td>
      <td>${d.disease||"—"}</td>
      <td>${d.year||"—"}</td>
      <td>${feat}</td>
      <td>${d.access_link?`<a href="${d.access_link}" target="_blank" style="color:var(--c-prep);font-size:0.77rem;">↗</a>`:"—"}</td>
    </tr>`;
  }).join("");
}
