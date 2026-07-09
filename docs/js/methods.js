// ── Data filters ──
let dFilter="All";
function initDataFilters() {
  const mods=[{k:"All",l:"All"},{k:"proteomics",l:"Spatial Proteomics"},{k:"transcriptomics",l:"Spatial Transcriptomics"}];
  document.getElementById("data-filters").innerHTML=mods.map(m=>
    `<button class="pill ${m.k==="All"?"active":""}" onclick="setDF('${m.k}',this)">${m.l}</button>`
  ).join("");
}
function setDF(v,btn) {
  dFilter=v;
  document.querySelectorAll("#data-filters .pill").forEach(p=>p.classList.remove("active"));
  btn.classList.add("active"); renderDatasets();
}


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


// ── Sort ──
function sortT(which,col) {
  if(which==="m") { if(mSort.col===col) mSort.dir*=-1; else{mSort.col=col;mSort.dir=1;} renderMethods(); }
  else            { if(dSort.col===col) dSort.dir*=-1; else{dSort.col=col;dSort.dir=1;} renderDatasets(); }
}


// ── Methods ──
function isComp(m) { return (m.category||"").toLowerCase().includes("computational"); }

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
  if(q) items=items.filter(d=>
    (d.id||"").toLowerCase().includes(q)||
    (d.internal_name||"").toLowerCase().includes(q)||
    (d.tissue||"").toLowerCase().includes(q)||
    (d.disease||"").toLowerCase().includes(q)||
    (d.organism||"").toLowerCase().includes(q)||
    (d.spatial_data_method||"").toLowerCase().includes(q)
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
