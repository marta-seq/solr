// ── Data filters ──
// Datasets tab is spatial-proteomics-only for now: the structured dataset
// registry's scope is SP-only (per CLAUDE.md), and ST rows are overwhelmingly
// missing disease/marker annotation (13/69 ST vs 62/72 SP have a disease
// value). Previously a toggle let users switch to All/ST; removed per
// explicit ask (2026-09-16) - only proteomics should appear, full stop, for
// now. Re-add a modality toggle if/when ST is brought properly into scope.

// ── Disease/tissue/platform/marker filters (canonicalized - see
// tissue_disease_maps.py, platform_maps.py, and 03_export_json.py's
// markers_list) ──
let diseaseFilter="All", tissueFilter="All", platformFilter="All";
// Markers are multi-select with AND semantics (a dataset must contain every
// selected marker, not just one) - e.g. "all pancreas datasets with CD3,
// CD45 and FOXP3" needs the intersection, not the union. 509 distinct
// markers exist in the real data, so this stays a <select> to pick from
// (browsable/searchable via native browser behavior) with chosen markers
// shown as removable chips below it, rather than rendering all as pills.
let markerFilters=[];

// Datasets tab is spatial-proteomics-only (see note at top of file) - every
// count/filter below is computed against this SP-scoped subset, not the
// full DATASETS array, so a pill's number always matches what the table
// could actually show.
function datasetsInScope() { return DATASETS.filter(d=>(d.spatial_data_category||"").toLowerCase().includes("proteomics")); }

// Applies every currently-active filter (disease/platform/tissue/markers/
// search box) to `items`. `exclude` skips one dimension's OWN filter
// ("disease"/"platform"/"tissue"/"marker") while still applying every
// other active one - this is what makes the pill/option counts faceted
// (e.g. "how many results if I also pick MIBI-TOF, given the disease/
// tissue/marker/search I already picked") instead of static global totals
// that don't reflect the current combination. Fixes the 2026-09-23 bug
// Marta found: static counts (e.g. "MIBI 22", "breast cancer 28") that
// didn't update together, so a combination showing 0 real results gave no
// warning from the pills themselves. Called with no `exclude` for the
// final rendered table and the results counter.
function applyDatasetFilters(items, exclude) {
  const q=(document.getElementById("data-search").value||"").toLowerCase();
  if(exclude!=="disease" && diseaseFilter!=="All") items=items.filter(d=>(d.disease_list||[]).includes(diseaseFilter));
  if(exclude!=="platform" && platformFilter!=="All") items=items.filter(d=>(d.platform_list||[]).includes(platformFilter));
  if(exclude!=="tissue" && tissueFilter!=="All") items=items.filter(d=>(d.tissue_list||[]).includes(tissueFilter));
  if(exclude!=="marker" && markerFilters.length) items=items.filter(d=>markerFilters.every(m=>(d.markers_list||[]).includes(m)));
  if(q) items=items.filter(d=>
    (d.id||"").toLowerCase().includes(q)||
    (d.internal_name||"").toLowerCase().includes(q)||
    (d.tissue||"").toLowerCase().includes(q)||
    (d.disease||"").toLowerCase().includes(q)||
    (d.organism||"").toLowerCase().includes(q)||
    (d.spatial_data_method||"").toLowerCase().includes(q)||
    (d.tissue_list||[]).some(t=>t.toLowerCase().includes(q))||
    (d.disease_list||[]).some(x=>x.toLowerCase().includes(q))||
    (d.disease_specifics_list||[]).some(x=>x.toLowerCase().includes(q))||
    (d.markers_list||[]).some(x=>x.toLowerCase().includes(q))
  );
  return items;
}

function countBy(items, listField) {
  const counts={};
  items.forEach(d=>(d[listField]||[]).forEach(v=>counts[v]=(counts[v]||0)+1));
  return counts;
}

function updateDiseaseFilters() {
  const base=applyDatasetFilters(datasetsInScope(),"disease");
  const counts=countBy(base,"disease_list");
  const diseases=Object.keys(counts).sort((a,b)=>counts[b]-counts[a]);
  document.getElementById("data-disease-filters").innerHTML=
    `<button class="pill ${diseaseFilter==="All"?"active":""}" onclick="setDiseaseFilter('All')">All (${base.length})</button>`+
    diseases.map(d=>
      `<button class="pill ${diseaseFilter===d?"active":""}" onclick="setDiseaseFilter('${d.replace(/'/g,"\\'")}')">${d} (${counts[d]})</button>`
    ).join("");
}
function setDiseaseFilter(v) { diseaseFilter=v; renderDatasets(); }

function updatePlatformFilters() {
  const base=applyDatasetFilters(datasetsInScope(),"platform");
  const counts=countBy(base,"platform_list");
  const platforms=Object.keys(counts).sort((a,b)=>counts[b]-counts[a]);
  document.getElementById("data-platform-filters").innerHTML=
    `<button class="pill ${platformFilter==="All"?"active":""}" onclick="setPlatformFilter('All')">All (${base.length})</button>`+
    platforms.map(p=>
      `<button class="pill ${platformFilter===p?"active":""}" onclick="setPlatformFilter('${p.replace(/'/g,"\\'")}')">${p} (${counts[p]})</button>`
    ).join("");
}
function setPlatformFilter(v) { platformFilter=v; renderDatasets(); }

// Both dropdowns are sorted by real usage frequency (most common tissue/
// marker first), not alphabetically.
function updateTissueFilter() {
  const base=applyDatasetFilters(datasetsInScope(),"tissue");
  const counts=countBy(base,"tissue_list");
  const entries=Object.entries(counts).sort((a,b)=>b[1]-a[1]);
  const sel=document.getElementById("data-tissue-filter");
  const prev=tissueFilter;
  sel.innerHTML=`<option value="All">All tissues (${base.length})</option>`+
    entries.map(([t,n])=>`<option value="${t}">${t} (${n})</option>`).join("");
  sel.value = counts[prev]!==undefined ? prev : "All";
  if (sel.value==="All" && prev!=="All") { tissueFilter="All"; }
}
function setTissueFilter(v) { tissueFilter=v; renderDatasets(); }

function updateMarkerFilter() {
  const base=applyDatasetFilters(datasetsInScope(),"marker");
  const counts=countBy(base,"markers_list");
  const entries=Object.entries(counts).sort((a,b)=>b[1]-a[1]);
  const sel=document.getElementById("data-marker-filter");
  sel.innerHTML=`<option value="All">Add a marker… (${base.length} datasets)</option>`+
    entries.map(([m,n])=>`<option value="${m}">${m} (${n})</option>`).join("");
  sel.value="All";
  renderMarkerChips();
}
function addMarkerFilter(v) {
  if (v!=="All" && !markerFilters.includes(v)) markerFilters.push(v);
  renderMarkerChips(); renderDatasets();
}
function removeMarkerFilter(v) {
  markerFilters=markerFilters.filter(m=>m!==v);
  renderMarkerChips(); renderDatasets();
}
function renderMarkerChips() {
  document.getElementById("data-marker-chips").innerHTML=markerFilters.map(m=>
    `<span class="chip">${m}<span class="chip-x" onclick="removeMarkerFilter('${m}')">×</span></span>`
  ).join("");
}

// Kept as thin aliases - index.html's onload calls these names once to do
// the first render; renderDatasets() itself calls the update* functions
// above on every subsequent filter/search change so counts stay live.
function initDiseaseFilters() { updateDiseaseFilters(); }
function initPlatformFilters() { updatePlatformFilters(); }
function initTissueFilter() { updateTissueFilter(); }
function initMarkerFilter() { updateMarkerFilter(); }


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
  if(which==="m")      { if(mSort.col===col) mSort.dir*=-1; else{mSort.col=col;mSort.dir=1;} renderMethods(); }
  else if(which==="a") { if(aSort.col===col) aSort.dir*=-1; else{aSort.col=col;aSort.dir=1;} renderApplications(); }
  else                 { if(dSort.col===col) dSort.dir*=-1; else{dSort.col=col;dSort.dir=1;} renderDatasets(); }
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
  if (col === "excel_order") {
    if (dir === -1) items.reverse();
  } else {
    items.sort((a,b)=>{
      const av=String(a[col]||""), bv=String(b[col]||"");
      if(col==="citations"||col==="year") return ((parseInt(av)||0)-(parseInt(bv)||0))*dir;
      return av.localeCompare(bv)*dir;
    });
  }
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
  // Refresh every filter's own pill/option counts against whatever's
  // currently active in every OTHER dimension - see applyDatasetFilters'
  // comment. Must run before building `items` below since it also
  // resets the marker <select>'s displayed value.
  updateDiseaseFilters(); updatePlatformFilters(); updateTissueFilter(); updateMarkerFilter();

  let items=applyDatasetFilters(datasetsInScope(), null);
  const resultCount=document.getElementById("data-results-count");
  if (resultCount) {
    const parts=[];
    if(diseaseFilter!=="All") parts.push(`disease: ${diseaseFilter}`);
    if(platformFilter!=="All") parts.push(`platform: ${platformFilter}`);
    if(tissueFilter!=="All") parts.push(`tissue: ${tissueFilter}`);
    if(markerFilters.length) parts.push(`markers: ${markerFilters.join(", ")}`);
    resultCount.textContent = parts.length
      ? `${items.length} dataset${items.length===1?"":"s"} match all selected filters (${parts.join(" · ")})`
      : `${items.length} dataset${items.length===1?"":"s"} total`;
  }
  const col=dSort.col, dir=dSort.dir;
  items.sort((a,b)=>String(a[col]||"").localeCompare(String(b[col]||""))*dir);
  const tb=document.getElementById("datasets-tbody");
  if(!items.length){tb.innerHTML=`<tr><td colspan="13" class="empty-state">No datasets found.</td></tr>`;return;}
  const MAX_MARKERS_SHOWN=4;
  tb.innerHTML=items.map(d=>{
    const isSP=(d.spatial_data_category||"").toLowerCase().includes("proteomics");
    const feat=d.n_markers||d.n_genes||"—";
    const markers=d.markers_list||[];
    const markersCell=markers.length
      ? `${markers.slice(0,MAX_MARKERS_SHOWN).join(", ")}${markers.length>MAX_MARKERS_SHOWN?` <span style="opacity:0.6;">+${markers.length-MAX_MARKERS_SHOWN}</span>`:""}`
      : "—";
    return `<tr>
      <td class="tid">${d.id}</td>
      <td class="tname">${d.internal_name||d.id}</td>
      <td>${isSP?`<span class="cbadge comp">SP</span>`:`<span class="cbadge data">ST</span>`}</td>
      <td>${d.spatial_data_method||"—"}</td>
      <td>${d.organism||"—"}</td>
      <td>${d.tissue||"—"}</td>
      <td>${d.disease||"—"}</td>
      <td>${d.n_patients||"—"}</td>
      <td>${d.n_images||"—"}</td>
      <td>${d.year||"—"}</td>
      <td>${feat}</td>
      <td style="font-size:0.77rem;" title="${markers.join(", ")}">${markersCell}</td>
      <td>${d.access_link?`<a href="${d.access_link}" target="_blank" style="color:var(--c-prep);font-size:0.77rem;">↗</a>`:"—"}</td>
    </tr>`;
  }).join("");
}
