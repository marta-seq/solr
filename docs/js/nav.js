// ── Nav ──
function showSection(name) {
  document.querySelectorAll(".main-section").forEach(s=>s.classList.remove("active"));
  document.querySelectorAll(".nav-links a").forEach(a=>a.classList.remove("active"));
  document.getElementById("section-"+name).classList.add("active");
  const n=document.getElementById("nav-"+name); if(n) n.classList.add("active");
  if(name==="graph"&&!cy) renderGraph();
  if(name==="book") loadChapter("index");
  window.scrollTo(0,0);
}
