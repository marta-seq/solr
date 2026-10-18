// ── Statistics tab ──
// Everything is computed client-side from METHODS / DATASETS (data/methods.json, data/datasets.json), so it never goes stale
// relative to the rest of the site. Papers are counted WITHOUT placeholders. Dates come from `pub_date` (publication date,
// YYYY / YYYY-MM / YYYY-MM-DD from Crossref) - NOT from when an entry was added to the review.
const ST_MODALITIES = [
  {key:"spatial_proteomics",     label:"Spatial proteomics",     color:"#388bfd"},
  {key:"spatial_transcriptomics",label:"Spatial transcriptomics",color:"#3fb950"},
  {key:"spatial_metabolomics",   label:"Spatial metabolomics",   color:"#ffa657"},
  {key:"multi_omics",            label:"Multi-omics",            color:"#d2a8ff"},
  {key:"other",                  label:"Other / not spatial omics", color:"#7d8590"},
  {key:"unspecified",            label:"Unspecified",            color:"#30363d"},
];
let statsMonth = null;

function stEsc(s) {
  return String(s==null?"":s).replace(/[&<>"']/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}
// "spatial_proteomics; spatial transcriptomics" -> ["spatial_proteomics","spatial_transcriptomics"]
function stModalities(raw) {
  const parts = String(raw||"").split(/[;,]/).map(x=>x.trim().toLowerCase().replace(/\s+/g,"_")).filter(Boolean);
  if (!parts.length) return ["unspecified"];
  const out = new Set();
  parts.forEach(p=>{
    if (["unknown","check","not_available","none","na","n/a"].includes(p)) out.add("unspecified");
    else if (["spatial_proteomics","spatial_transcriptomics","spatial_metabolomics","multi_omics"].includes(p)) out.add(p);
    else out.add("other");
  });
  return [...out];
}
function stPapers() { return METHODS.filter(m=>!m.is_placeholder); }
function stIsMethod(m) { return m.paper_type==="method"; }
function stCats(m) {
  const c = (m.pipeline_categories||[]).map(x=>String(x).trim()).filter(Boolean);
  return c.length ? c : ["Uncategorized"];
}
function stTodayStr() { return new Date().toISOString().slice(0,10); }
function stMonthOf(m) {            // "YYYY-MM" or null; never in the future
  const d = m.pub_date||""; if (d.length<7) return null;
  const mo = d.slice(0,7); return mo > stTodayStr().slice(0,7) ? null : mo;
}
function stYearOf(m) {
  const y = parseInt((m.pub_date||"").slice(0,4) || m.year, 10);
  return (y>=1990 && y<=new Date().getFullYear()) ? y : null;
}
function stCount(items, keyFn) {
  const out = {}; items.forEach(it=>keyFn(it).forEach(k=>{ out[k]=(out[k]||0)+1; })); return out;
}
function stBars(rows, opts) {      // rows: [{label, value, color, sub}]  -> horizontal bars as HTML
  const max = Math.max(1, ...rows.map(r=>r.value));
  return rows.map(r=>`<div class="st-row"><div class="st-lab" title="${stEsc(r.label)}">${stEsc(r.label)}</div>
    <div class="st-track"><div class="st-fill" style="width:${(100*r.value/max).toFixed(1)}%;background:${r.color||"#388bfd"}"></div></div>
    <div class="st-val">${r.value}${r.sub?` <span class="st-sub">${stEsc(r.sub)}</span>`:""}</div></div>`).join("");
}
function stStackedColumns(labels, series, height) {   // series: [{name,color,values[]}] -> inline SVG
  const W = Math.max(520, labels.length*34), H = height||240, pad = {l:36,r:8,t:10,b:34};
  const totals = labels.map((_,i)=>series.reduce((s,x)=>s+x.values[i],0));
  const max = Math.max(1, ...totals), bw = (W-pad.l-pad.r)/labels.length;
  let svg = `<svg viewBox="0 0 ${W} ${H}" class="st-svg" role="img">`;
  for (let g=0; g<=4; g++) {
    const v = Math.round(max*g/4), y = H-pad.b-(H-pad.t-pad.b)*g/4;
    svg += `<line x1="${pad.l}" x2="${W-pad.r}" y1="${y}" y2="${y}" stroke="#21262d"/><text x="${pad.l-6}" y="${y+3}" text-anchor="end" class="st-axis">${v}</text>`;
  }
  labels.forEach((lab,i)=>{
    let y = H-pad.b;
    series.forEach(s=>{
      const h = (H-pad.t-pad.b)*s.values[i]/max;
      if (h>0) { y -= h; svg += `<rect x="${pad.l+i*bw+bw*0.15}" y="${y}" width="${bw*0.7}" height="${h}" fill="${s.color}"><title>${stEsc(lab)} - ${stEsc(s.name)}: ${s.values[i]}</title></rect>`; }
    });
    if (labels.length<=16 || i%2===0) svg += `<text x="${pad.l+i*bw+bw/2}" y="${H-pad.b+14}" text-anchor="middle" class="st-axis">${stEsc(lab)}</text>`;
  });
  return svg + `</svg>` + `<div class="st-legend">${series.map(s=>`<span><i style="background:${s.color}"></i>${stEsc(s.name)}</span>`).join("")}</div>`;
}
function stDelta(cur, prev) {
  if (!prev && !cur) return `<span class="st-sub">–</span>`;
  const d = cur-prev, cls = d>0?"st-up":d<0?"st-down":"st-sub";
  return `<span class="${cls}">${d>0?"▲ +":d<0?"▼ ":"= "}${d===0?"":d}</span> <span class="st-sub">vs ${prev}</span>`;
}

function renderStats() {
  const el = document.getElementById("stats-body"); if (!el) return;
  const papers = stPapers(), methods = papers.filter(stIsMethod), apps = papers.filter(m=>!stIsMethod(m));

  // 1. headline numbers
  const kpi = (n,l,c)=>`<div class="stat-cell"><span class="stat-number" style="color:${c}">${n}</span><span class="stat-label">${l}</span></div>`;
  const dsMod = stCount(DATASETS, d=>stModalities(d.spatial_data_category));
  const paperMod = stCount(papers, m=>stModalities(m.spatial_data_category));
  let html = `<div class="stats-bar st-kpis">${kpi(papers.length,"Papers","#388bfd")}${kpi(methods.length,"Methods","#3fb950")}
    ${kpi(apps.length,"Applications","#d2a8ff")}${kpi(DATASETS.length,"Datasets","#ffa657")}</div>`;

  // 2. by modality (papers and datasets)
  const modRows = (counts)=>stBars(ST_MODALITIES.map(m=>({label:m.label, value:counts[m.key]||0, color:m.color})));
  html += `<div class="st-grid"><div class="st-card"><h3>Papers by spatial modality</h3>${modRows(paperMod)}
      <p class="st-note">A paper can have several modalities, so the bars add up to more than the paper total.</p></div>
    <div class="st-card"><h3>Datasets by spatial modality</h3>${modRows(dsMod)}</div></div>`;

  // 3. papers per year
  const years = papers.map(stYearOf).filter(Boolean);
  if (years.length) {
    const y0 = Math.max(Math.min(...years), 2010), y1 = Math.max(...years), labels = [];
    for (let y=y0; y<=y1; y++) labels.push(String(y));
    const per = (list)=>labels.map(l=>list.filter(m=>stYearOf(m)===parseInt(l,10)).length);
    html += `<div class="st-card"><h3>Papers per year</h3>${stStackedColumns(labels,
      [{name:"Methods",color:"#3fb950",values:per(methods)},{name:"Applications",color:"#d2a8ff",values:per(apps)}])}
      <p class="st-note">By publication year (Crossref). ${papers.length-years.length} papers have no usable date; papers before ${y0} are not shown.</p></div>`;
  }

  // 4. methods per pipeline category
  const catCounts = stCount(methods, stCats);
  const catRows = Object.entries(catCounts).sort((a,b)=>b[1]-a[1]).slice(0,22)
    .map(([c,n])=>({label:c, value:n, color: c==="Uncategorized" ? "#30363d" : catInfo(c).color}));
  html += `<div class="st-card"><h3>Methods per pipeline category</h3>${stBars(catRows)}
    <p class="st-note">A method can sit in several categories. Top ${catRows.length} of ${Object.keys(catCounts).length} shown.</p></div>`;

  // 5. trends
  html += `<div class="st-card" id="st-trends"><h3>Trends</h3><div id="st-trends-body"></div></div>`;
  el.innerHTML = html;
  renderStatsTrends();
}

function renderStatsTrends() {
  const body = document.getElementById("st-trends-body"); if (!body) return;
  const papers = stPapers(), withMonth = papers.filter(m=>stMonthOf(m));
  const months = [...new Set(withMonth.map(stMonthOf))].sort().reverse().slice(0,24);
  if (!months.length) { body.innerHTML = `<p class="st-note">No papers with a month-level publication date yet.</p>`; return; }
  const nowMonth = stTodayStr().slice(0,7);
  if (!statsMonth || !months.includes(statsMonth)) statsMonth = months.find(m=>m<nowMonth) || months[0];   // default: last COMPLETE month
  const prevIdx = months.indexOf(statsMonth)+1, prevMonth = months[prevIdx] || null;
  const inM = (mo)=>papers.filter(m=>stMonthOf(m)===mo);
  const cur = inM(statsMonth), prev = prevMonth ? inM(prevMonth) : [];
  const nm = (l)=>l.filter(stIsMethod).length, na = (l)=>l.filter(m=>!stIsMethod(m)).length;

  // last 12 months timeline
  const tl = months.slice(0,12).reverse();
  const timeline = stStackedColumns(tl, [
    {name:"Methods", color:"#3fb950", values: tl.map(mo=>nm(inM(mo)))},
    {name:"Applications", color:"#d2a8ff", values: tl.map(mo=>na(inM(mo)))}], 200);

  const catC = stCount(cur.filter(stIsMethod), stCats), catP = stCount(prev.filter(stIsMethod), stCats);
  const catRows = Object.entries(catC).sort((a,b)=>b[1]-a[1]).slice(0,8)
    .map(([c,n])=>({label:c, value:n, color: c==="Uncategorized" ? "#30363d" : catInfo(c).color, sub: prevMonth ? `(prev ${catP[c]||0})` : ""}));
  const modC = stCount(cur, m=>stModalities(m.spatial_data_category)), modP = stCount(prev, m=>stModalities(m.spatial_data_category));
  const modRows = ST_MODALITIES.filter(m=>modC[m.key]||modP[m.key]).map(m=>({label:m.label, value:modC[m.key]||0, color:m.color, sub: prevMonth ? `(prev ${modP[m.key]||0})` : ""}));

  const list = cur.slice().sort((a,b)=>(b.pub_date||"").localeCompare(a.pub_date||"")).slice(0,40).map(m=>{
    const href = m.doi ? (m.doi.startsWith("http") ? m.doi : "https://doi.org/"+m.doi) : null;
    const t = stEsc(m.title||m.name||m.id);
    return `<li>${href?`<a href="${stEsc(href)}" target="_blank" rel="noopener">${t}</a>`:t}
      <span class="st-sub">${stIsMethod(m)?"method":"application"} · ${stEsc(m.pub_date||"")}${m.journal?" · "+stEsc(m.journal):""}</span></li>`;
  }).join("");

  body.innerHTML = `
    <div class="filter-row"><span class="st-sub">Month (by publication date):</span>
      <select class="select-filter" onchange="statsMonth=this.value;renderStatsTrends()">${months.map(mo=>`<option value="${mo}" ${mo===statsMonth?"selected":""}>${mo}${mo===nowMonth?" (in progress)":""}</option>`).join("")}</select></div>
    <div class="stats-bar st-kpis">
      <div class="stat-cell"><span class="stat-number">${cur.length}</span><span class="stat-label">Papers</span><div class="st-d">${prevMonth?stDelta(cur.length,prev.length):""}</div></div>
      <div class="stat-cell"><span class="stat-number" style="color:#3fb950">${nm(cur)}</span><span class="stat-label">Methods</span><div class="st-d">${prevMonth?stDelta(nm(cur),nm(prev)):""}</div></div>
      <div class="stat-cell"><span class="stat-number" style="color:#d2a8ff">${na(cur)}</span><span class="stat-label">Applications</span><div class="st-d">${prevMonth?stDelta(na(cur),na(prev)):""}</div></div>
    </div>
    <div class="st-grid"><div><h4>Method categories this month</h4>${catRows.length?stBars(catRows):`<p class="st-note">No methods this month.</p>`}</div>
      <div><h4>Modalities this month</h4>${modRows.length?stBars(modRows):`<p class="st-note">No papers this month.</p>`}</div></div>
    <h4>Last 12 months</h4>${timeline}
    <h4>Papers published in ${statsMonth}${cur.length>40?" (latest 40)":""}</h4><ul class="st-list">${list||"<li class='st-sub'>none</li>"}</ul>
    <p class="st-note">Counts are what the review contains, not everything published: the most recent months are incomplete until the weekly scan and
    review catch up, so compare months with care.</p>`;
}
