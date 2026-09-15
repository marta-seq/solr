// ── Applications ──
// Application papers (paper_type === "application") have no pipeline
// category, so they get their own table + their own graph (positioned by
// title/abstract embedding similarity, see 04_compute_embeddings.py),
// rather than being folded into the Methods table/graph which assume a
// category exists.
function isApp(m) { return m.paper_type === "application"; }

let aSort={col:"excel_order",dir:1};
let appModalityFilter="All";

function initAppFilters() {
  const counts={};
  METHODS.filter(m=>!m.is_placeholder&&isApp(m)).forEach(m=>{
    (m.spatial_modality_ap||[]).forEach(mod=>{ counts[mod]=(counts[mod]||0)+1; });
  });
  const entries=Object.entries(counts).sort((a,b)=>b[1]-a[1]);
  const pills=[{k:"All",l:"All"},...entries.map(([mod,n])=>({k:mod,l:`${mod.replace(/^spatial_/,"").replace(/_/g," ")} (${n})`}))];
  const html=pills.map(p=>
    `<button class="pill ${p.k==="All"?"active":""}" onclick="setAppModalityFilter('${p.k}',this,'app-modality-filters')">${p.l}</button>`
  ).join("");
  document.getElementById("app-modality-filters").innerHTML=html;
  document.getElementById("app-graph-modality-filters").innerHTML=html.replaceAll("app-modality-filters","app-graph-modality-filters");
}
function setAppModalityFilter(v,btn,rowId) {
  appModalityFilter=v;
  document.querySelectorAll(`#${rowId} .pill`).forEach(p=>p.classList.remove("active"));
  btn.classList.add("active");
  if (rowId==="app-modality-filters") renderApplications(); else renderAppGraph();
}

// ── Applications table ──
function renderApplications() {
  const q=(document.getElementById("app-search").value||"").toLowerCase();
  let items=METHODS.filter(m=>!m.is_placeholder&&isApp(m));
  if(appModalityFilter!=="All") items=items.filter(m=>(m.spatial_modality_ap||[]).includes(appModalityFilter));
  if(q) items=items.filter(m=>
    (m.title||"").toLowerCase().includes(q)||
    (m.id||"").toLowerCase().includes(q)||
    (m.journal||"").toLowerCase().includes(q)
  );
  const col=aSort.col, dir=aSort.dir;
  if (col === "excel_order") {
    if (dir === -1) items.reverse();
  } else {
    items.sort((a,b)=>{
      const av=String(a[col]||""), bv=String(b[col]||"");
      if(col==="year") return ((parseInt(av)||0)-(parseInt(bv)||0))*dir;
      return av.localeCompare(bv)*dir;
    });
  }
  const tb=document.getElementById("apps-tbody");
  if(!items.length){tb.innerHTML=`<tr><td colspan="7" class="empty-state">No application papers found.</td></tr>`;return;}
  tb.innerHTML=items.map(m=>{
    const dsIds=(m.associated_data_ids||[]);
    const mod=(m.spatial_modality_ap||[]).map(x=>x.replace(/^spatial_/,"")).join(", ")||"—";
    return `<tr onclick="openInAppGraph('${m.id}')">
      <td class="tid">${m.id}</td>
      <td class="tname">${m.title||m.id}</td>
      <td>${m.year||"—"}</td>
      <td>${m.journal?m.journal.slice(0,28)+(m.journal.length>28?"…":""):"—"}</td>
      <td>${mod}</td>
      <td>${dsIds.length?`<span class="cbadge data">${dsIds.length}</span>`:"—"}</td>
      <td><span class="sdot ${m.review_status==="manual"?"manual":""}" title="${m.review_status||"stub"}"></span></td>
    </tr>`;
  }).join("");
}

// ── Applications graph ──
// No categories to anchor positions with (unlike the Methods graph's
// per-category sunflower spiral), so position comes only from
// EMBEDDINGS[id] = [x,y] (see 04_compute_embeddings.py). If embeddings.json
// hasn't been generated yet (fetch 404s in data.js -> EMBEDDINGS={}), falls
// back to a plain grid so the graph still renders something rather than
// collapsing every node onto (0,0).
let appCy=null;
let appShowData=true;

function renderAppGraph() {
  const apps=METHODS.filter(m=>!m.is_placeholder&&isApp(m))
    .filter(m=>appModalityFilter==="All"||(m.spatial_modality_ap||[]).includes(appModalityFilter));
  // 04_compute_embeddings.py already scales/centers coordinates to a
  // target spread, so no extra multiplier needed here - use raw values.
  const EMBED_SCALE=1;
  const GRID_COLS=20, GRID_STEP=55;
  const elements=[];
  const appPos={};

  apps.forEach((m,i)=>{
    const emb=EMBEDDINGS[m.id];
    let px,py;
    if (emb) { px=emb[0]*EMBED_SCALE; py=emb[1]*EMBED_SCALE; }
    else     { px=(i%GRID_COLS)*GRID_STEP; py=Math.floor(i/GRID_COLS)*GRID_STEP; }
    appPos[m.id]={x:px,y:py};

    const mods=m.spatial_modality_ap||[];
    const isSP=mods.includes("spatial_proteomics"), isST=mods.includes("spatial_transcriptomics");
    const color = isSP&&isST ? "#d29922" : isSP ? "#3fb950" : isST ? "#a371f7" : "#8b949e";
    const label = m.title ? (m.title.length>40 ? m.title.slice(0,40)+"…" : m.title) : m.id;
    elements.push({
      data:{id:m.id, label, color, type:"app", _a:JSON.stringify(m)},
      position:{x:px,y:py}
    });
  });

  if (appShowData) {
    const usedIds=new Set();
    apps.forEach(m=>(m.associated_data_ids||[]).forEach(d=>{ if(d) usedIds.add(d); }));
    const dsMap={};
    DATASETS.forEach(d=>{ if(usedIds.has(d.id)) dsMap[d.id]=d; });
    const dsLinks={};
    apps.forEach(m=>(m.associated_data_ids||[]).forEach(did=>{
      if(did && dsMap[did]) (dsLinks[did]=dsLinks[did]||[]).push(m.id);
    }));
    Object.entries(dsMap).forEach(([id,d],i)=>{
      const linkedIds=dsLinks[id]||[];
      const pts=linkedIds.map(mid=>appPos[mid]).filter(Boolean);
      const cx=pts.length?pts.reduce((s,p)=>s+p.x,0)/pts.length:0;
      const cyy=pts.length?pts.reduce((s,p)=>s+p.y,0)/pts.length:0;
      const angle=i*2.399, r=15;
      elements.push({
        data:{id:"ds_"+d.id, label:d.internal_name||d.id, type:"dataset", _d:JSON.stringify(d)},
        position:{x:cx+r*Math.cos(angle), y:cyy+r*Math.sin(angle)}
      });
    });
    apps.forEach(m=>(m.associated_data_ids||[]).forEach(did=>{
      if(did && dsMap[did]) elements.push({data:{id:`ae_${m.id}_${did}`, source:m.id, target:"ds_"+did, type:"data-edge"}});
    }));
  }

  if(appCy) appCy.destroy();
  appCy = cytoscape({
    container: document.getElementById("appgraph-container"),
    elements,
    style: [
      { selector: "node[type='app']", style: {
        "background-color":"data(color)", "width":12, "height":12,
        "label":"data(label)", "color":"#e6edf3", "font-size":"8px", "font-family":"Inter,sans-serif",
        "text-valign":"bottom", "text-margin-y":3,
        "text-outline-width":1.2, "text-outline-color":"#0d1117",
        "border-width":1, "border-color":"#21262d",
      }},
      { selector: "node[type='dataset']", style: {
        "background-color":"#0d1117", "shape":"round-rectangle",
        "border-width":2, "border-style":"dashed", "border-color":"#39d353",
        "width":10, "height":10, "label":"data(label)", "color":"#e6edf3",
        "font-size":"7px", "font-family":"JetBrains Mono,monospace",
        "text-valign":"bottom", "text-margin-y":2,
        "text-outline-width":1, "text-outline-color":"#0d1117",
      }},
      { selector: "edge[type='data-edge']", style: {
        "line-color":"#39d353", "line-style":"dashed", "width":1, "opacity":0.35, "curve-style":"bezier",
      }},
      { selector: "node:selected", style: { "border-width":3, "border-color":"#fff" } },
    ],
    layout:{name:"preset", animate:false},
    userZoomingEnabled:true, userPanningEnabled:true,
  });
  appCy.fit(appCy.nodes(), 40);

  appCy.on("tap", "node", evt=>{
    const d=evt.target.data();
    try {
      if(d.type==="app" && d._a) showAppNodeDetail(JSON.parse(d._a));
      else if(d.type==="dataset" && d._d) showAppDsDetail(JSON.parse(d._d));
    } catch(e){ console.error(e); }
  });
  appCy.on("tap", evt=>{ if(evt.target===appCy) closeAppDetail(); });
}

function toggleAppData() {
  appShowData=document.getElementById("tog-appdata").checked;
  renderAppGraph();
}

function showAppNodeDetail(m) {
  document.getElementById("ad-id").textContent=m.id;
  document.getElementById("ad-name").textContent=m.title||m.id;
  document.getElementById("ad-sub").textContent=[m.year,m.journal].filter(Boolean).join(" · ");
  document.getElementById("ad-abs").textContent=(m.spatial_modality_ap||[]).map(x=>x.replace(/^spatial_/,"")).join(", ")||"";
  document.getElementById("ad-doi").innerHTML=m.doi?`<a href="${m.doi}" target="_blank">${m.doi}</a>`:"";
  const dsIds=m.associated_data_ids||[];
  let c="";
  if (dsIds.length) {
    c+=`<strong>Datasets (${dsIds.length})</strong>`;
    c+=dsIds.map(id=>{ const ds=DATASETS.find(x=>x.id===id); return `<div>· ${ds?(ds.internal_name||id):id}</div>`; }).join("");
  }
  document.getElementById("ad-conns").innerHTML=c;
  document.getElementById("appdpanel").classList.add("on");
}

function showAppDsDetail(d) {
  document.getElementById("ad-id").textContent=d.id;
  document.getElementById("ad-name").textContent=d.internal_name||d.id;
  document.getElementById("ad-sub").textContent=[d.spatial_data_method,d.tissue,d.disease,d.organism].filter(Boolean).join(" · ");
  document.getElementById("ad-abs").textContent=d.notes||"";
  document.getElementById("ad-doi").innerHTML=d.access_link?`<a href="${d.access_link}" target="_blank">↗ Access data</a>`:"";
  document.getElementById("ad-conns").innerHTML="";
  document.getElementById("appdpanel").classList.add("on");
}

function closeAppDetail() { document.getElementById("appdpanel").classList.remove("on"); }

function openInAppGraph(id) {
  showSection("appgraph");
  setTimeout(()=>{
    if(!appCy) return;
    const node=appCy.getElementById(id);
    if(node.length){
      appCy.animate({fit:{eles:node,padding:100},duration:400});
      const m=METHODS.find(x=>x.id===id);
      if(m) showAppNodeDetail(m);
    }
  },500);
}
