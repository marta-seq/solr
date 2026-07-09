// ── State ──
let METHODS=[], DATASETS=[], STATS={}, cy=null;
let mSort={col:"name",dir:1}, dSort={col:"id",dir:1};
let gStageFilter="All";

const CATS = {
  "Preprocessing":                              {color:"#388bfd", x:120,  y:100},
  "Cell segmentation - Imaging based":          {color:"#3fb950", x:340,  y:80},
  "Cell segmentation - Transcript based":       {color:"#56d364", x:340,  y:220},
  "Phenotyping":                                {color:"#ffa657", x:560,  y:80},
  "Niche/Neighborhood/domain analysis":         {color:"#d2a8ff", x:560,  y:260},
  "Spatial Variable Genes":                     {color:"#79c0ff", x:780,  y:100},
  "Cell-Cell-Communication":                    {color:"#ff7b72", x:780,  y:300},
  "Label separation/pattern extraction - ML":   {color:"#f0883e", x:560,  y:450},
  "General Framework":                          {color:"#8b949e", x:120,  y:380},
  "Cell type Deconvolution":                    {color:"#e3b341", x:340,  y:400},
};

function catInfo(cat) {
  if (!cat) return {color:"#8b949e", x:450, y:280};
  for (const [k,v] of Object.entries(CATS)) {
    if (cat.toLowerCase().includes(k.toLowerCase())) return v;
  }
  return {color:"#8b949e", x:450, y:280};
}
