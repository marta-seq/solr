// ── State ──
let METHODS=[], DATASETS=[], STATS={}, cy=null;
let mSort={col:"name",dir:1}, dSort={col:"id",dir:1};
let gStageFilter="All";

// Centroids are in a ~1400x800 virtual canvas
// Layout mirrors the pipeline flow: left=upstream, right=downstream
// Segmentation close to preprocessing, niche close to segmentation
const CATS = {
  "Preprocessing":                              {color:"#388bfd", x:180,  y:200},
  "Cell segmentation - Imaging based":          {color:"#3fb950", x:420,  y:130},
  "Cell segmentation - Transcript based":       {color:"#56d364", x:420,  y:310},
  "Phenotyping":                                {color:"#ffa657", x:660,  y:130},
  "Niche/Neighborhood/domain analysis":         {color:"#d2a8ff", x:660,  y:360},
  "Spatial Variable Genes":                     {color:"#79c0ff", x:900,  y:130},
  "Cell-Cell-Communication":                    {color:"#ff7b72", x:900,  y:380},
  "Label separation/pattern extraction - ML":   {color:"#f0883e", x:660,  y:580},
  "General Framework":                          {color:"#8b949e", x:180,  y:500},
  "Cell type Deconvolution":                    {color:"#e3b341", x:420,  y:530},
  "Subcellular localization":                   {color:"#bc8cff", x:900,  y:580},
  "Integration of modalities":                  {color:"#c9d1d9", x:1100, y:280},
  "spatiotemporal dynamics":                    {color:"#f78166", x:1100, y:480},
  // Added 2026-09-01 alongside the pipeline_category taxonomy cleanup
  // (see src/preprocessing/category_maps.py) - without an entry here, any
  // method mapped to one of these fell into the generic gray default at
  // canvas center instead of getting a real color/position.
  "Cell segmentation - unspecified":            {color:"#58a6ff", x:420,  y:220},
  "Clustering":                                 {color:"#a5d6ff", x:660,  y:250},
  "Survival prediction":                        {color:"#ffab70", x:1250, y:400},
  "Data alignment / integration / imputation":  {color:"#39c5cf", x:1100, y:150},
  "Foundation model":                           {color:"#d29922", x:300,  y:650},
  "Computer vision (H&E)":                      {color:"#ff9bce", x:550,  y:50},
  "Virtual staining (proteomics)":              {color:"#f2cc60", x:700,  y:50},
  "Virtual staining (transcriptomics)":         {color:"#56d4dd", x:850,  y:50},
  "Immune infiltration scoring":                {color:"#e0729c", x:900,  y:250},
  "Analysis/workflow optimization":             {color:"#768390", x:60,   y:350},
  "Other":                                      {color:"#8b949e", x:1300, y:700},
};

function catInfo(cat) {
  if (!cat) return {color:"#8b949e", x:640, y:350};
  const cl = cat.toLowerCase();
  for (const [k,v] of Object.entries(CATS)) {
    if (cl.includes(k.toLowerCase())) return v;
  }
  return {color:"#8b949e", x:640, y:350};
}
