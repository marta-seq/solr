// ── Load ──
async function loadData() {
  try {
    const base = window.location.pathname.includes('/solr/') ? '/solr/' : './';
    const [m,d,s] = await Promise.all([
      fetch(base+"data/methods.json").then(r=>{if(!r.ok)throw new Error("methods.json "+r.status);return r.json();}),
      fetch(base+"data/datasets.json").then(r=>{if(!r.ok)throw new Error("datasets.json "+r.status);return r.json();}),
      fetch(base+"data/stats.json").then(r=>{if(!r.ok)throw new Error("stats.json "+r.status);return r.json();}),
    ]);
    METHODS=m; DATASETS=d; STATS=s;
    initStats(); initDataFilters(); initDiseaseFilters(); initTissueFilter(); initMarkerFilter();
    initGraphFilters(); initSourceFilters();
    buildLegend(); renderMethods(); renderDatasets();
  } catch(e) { console.error("Load failed:",e); alert("Data load failed: "+e.message); }
}


// ── Stats ──
function animN(el,n) {
  n=parseInt(n)||0; let cur=0, step=Math.max(1,Math.ceil(n/30));
  const t=setInterval(()=>{ cur=Math.min(cur+step,n); el.textContent=cur; if(cur>=n)clearInterval(t); },28);
}
function initStats() {
  animN(document.getElementById("stat-papers"),  STATS.total_papers);
  animN(document.getElementById("stat-methods"), STATS.comp_methods);
  animN(document.getElementById("stat-curated"), STATS.curated);
  animN(document.getElementById("stat-datasets"),STATS.total_datasets);
  animN(document.getElementById("stat-sp"),      STATS.sp_datasets);
  animN(document.getElementById("stat-st"),      STATS.st_datasets);
  const mc=document.getElementById("card-mc"); if(mc) mc.textContent=STATS.comp_methods;
  const dc=document.getElementById("card-dc"); if(dc) dc.textContent=STATS.total_datasets;
}
