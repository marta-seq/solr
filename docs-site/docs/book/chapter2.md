# Background {#ch:background}

### Approaches to Studying Cancer

Understanding the complexity of cancer, particularly its cellular
heterogeneity and dynamic microenvironment, relies heavily on advanced
analytical technologies.

Traditionally, cancer research has been dominated by genomic and
transcriptomic approaches, which have illuminated key mutations,
epigenetic alterations, and dysregulated pathways across tumour
types[@B16_TCGA2013; @B17_2018; @B18_TCGA2012; @B19_nunes2024]. More
recently, multi-omics strategies---including
epigenomics[@B20_li; @B21_chen_mapping_2025; @B22_baylin2011],
proteomics[@B23_protmap2015; @B24_suhre2021; @B25_aebersold2016], and
metabolomics[@B26_schmidt2021] --- have expanded our understanding of
tumour biology by capturing additional layers of molecular regulation.

Single-cell technologies have further revolutionized the field by
enabling the dissection of intra-tumoral heterogeneity at cellular
resolution, uncovering rare cell populations and lineage trajectories
that bulk analyses obscure[@B27_aldridge2020; @B28_kinker2020]. These
innovations have also extended to single-cell
proteomics[@B29_mansuri2023; @B30_protchalg2023; @B31_setayesh2023] and
metabolomics[@B32_seydel_metab_2021; @B33_spatial_metab2023], offering
increasingly comprehensive insights into tumour ecosystems. However,
despite their power, these methods often lack spatial context---an
essential component for understanding how cellular localization and
cell--cell interactions within the tumour microenvironment influence
disease progression and treatment response.

On the other hand, spatial information is routinely obtained in the
clinic through histopathological staining (e.g., H&E),
[IHC]{acronym-label="IHC" acronym-form="singular+short"}, and
radiological imaging[@B34_TCIA2013]. These approaches offer spatial
resolution at the tissue or organ level and remain vital for diagnosis
and clinical decision-making, but are inherently limited in molecular
detail and throughput, lacking the capacity to resolve cellular
heterogeneity or dynamic molecular interactions at scale.

To bridge this gap, spatial omics technologies have emerged as powerful
tools that integrate high-resolution molecular profiling with spatial
localization. These approaches allow the mapping of transcripts,
proteins, and metabolites directly within their tissue
context---preserving the native architecture of tumours and their
microenvironments.

## Spatially Resolved Omics Techniques

Spatial omics technologies have the potential to advance our
understanding of tumour ecosystems and improve clinical outcomes. The
essence of spatial omics lies in its aptitude for the simultaneous
detection of molecular constituents at exact spatial
coordinates[@B35_liu_spatiotemporal2024]. However different techniques
vary greatly in resolution, scale and molecular
complexity[@B36_palla_spatial_2022].

### Spatial proteomics

Spatial proteomics encompasses technologies that enable the in-situ
profiling of proteins within tissues, preserving their spatial
localization. Most spatial proteomics techniques detect proteins using
antibodies tagged with fluorophores, metals, or DNA barcodes. These tags
are then read using technologies such as fluorescence microscopy, mass
spectrometry, or DNA-based imaging to map the spatial distribution of
proteins within
tissues[@B35_liu_spatiotemporal2024; @B37_elundberg2019; @B38_G1_jing2025; @B39_hickey2022].
A summary of key features across these technologies is provided in
Table [1.1](#tab:spatial_proteomics){reference-type="ref"
reference="tab:spatial_proteomics"}.

Immunohistochemistry (IHC) is one of the most established clinical tools
for protein detection[@B40_IHC2019]. It uses enzyme-linked antibodies to
generate a chromogenic signal visible under brightfield microscopy.
While widely available and routinely used in diagnostics, traditional
IHC is generally limited to detecting one or a few markers per tissue
section, making it unsuitable for high-dimensional spatial profiling.

Among the fluorescence-based approaches, [IF]{acronym-label="IF"
acronym-form="singular+short"}[@B41_ijsselsteijn2019] is widely used but
limited in multiplexing. More advanced cyclic immunofluorescence
methods---such as [t-CyCIF]{acronym-label="t-CyCIF"
acronym-form="singular+short"}[@B42_Cycif2018] and
[IBEX]{acronym-label="IBEX"
acronym-form="singular+short"}[@B43_ibex_2022] - use iterative staining
and imaging cycles to overcome spectral limitations, enabling the
detection of 40--60+ markers with spatial resolution down to 200--300
nm.

DNA-barcoded approaches such as [CODEX]{acronym-label="CODEX"
acronym-form="singular+short"}[@P5_B44_S2_CODEX2018] and
[Immuno-SABER]{acronym-label="Immuno-SABER"
acronym-form="singular+short"}[@B45_isaber2019] further enhance
multiplexing. CODEX utilizes DNA-barcoded antibodies and sequential
hybridization of fluorescent probes, achieving high multiplexing (up to
60 proteins) with single-cell resolution ($\sim$`<!-- -->`{=html}500 nm)
in a single imaging plane[@B46_codex]. Immuno-SABER employs orthogonal
DNA concatemers for signal amplification achieving 32-plex with same
resolution as COSMX[@B45_isaber2019].

Mass spectrometry-based approaches---notably [IMC]{acronym-label="IMC"
acronym-form="singular+short"}[@B47_B72_S4_IMC2014] and
[MIBI-TOF]{acronym-label="MIBI-TOF"
acronym-form="singular+short"}[@B48_P4_keren_mibi-tof_2019]---use
antibodies conjugated to isotopically pure lanthanide metals, which are
detected using laser ablation (IMC) or ion beams (MIBI-TOF). These
methods avoid fluorescence background and allow simultaneous
quantification of 40--50 proteins per tissue section. IMC offers spatial
resolution of approximately 1 μm, while MIBI achieves higher resolution
($\sim$`<!-- -->`{=html}300 nm), albeit with more complex
instrumentation.

::: {#tab:spatial_proteomics}
  ------------------- ----------------------------- ------------------------------------ ------------- -- --
                                                                                                          
  **Type**                                                                                                
  **Capacity**                                                                                            
  **Scale**                                                                                               
  IHC                                                                                                     
  (chromogen)                                                                                             
  Microscopy          1--2                          $\sim$`<!-- -->`{=html}1--5 µm         Cellular       
  IF                  Fluorophore                                                                         
  Microscopy          4--7                          $\sim$`<!-- -->`{=html}200--500 nm    Subcellular     
  cyCIF / 4i / IBEX                                                                                       
  (iterative)                                                                                             
  Imaging             $>$`<!-- -->`{=html}60        $\sim$`<!-- -->`{=html}250 nm         Subcellular     
  CODEX               DNA Barcode                                                                         
  Microscopy          $>$`<!-- -->`{=html}40--60+   $\sim$`<!-- -->`{=html}250 nm         Subcellular     
  Immuno-SABER                                                                                            
  (concatemers)                                                                                           
  Microscopy          $>$`<!-- -->`{=html}50+       $\sim$`<!-- -->`{=html}200--300 nm    Subcellular     
  IMC                 Metal Isotopes                                                                      
  \+ MS               $\sim$`<!-- -->`{=html}40     $\sim$`<!-- -->`{=html}1 µm            Cellular       
  MIBI                Metal Isotopes                                                                      
  \+ MS               $\sim$`<!-- -->`{=html}40     $\sim$`<!-- -->`{=html}300 nm         Subcellular     
  ------------------- ----------------------------- ------------------------------------ ------------- -- --

  : Comparison table of key features of spatial proteomics profiling
  methods.
:::

Together, these spatial proteomics platforms offer complementary
advantages in terms of marker throughput, resolution, and sensitivity,
enabling comprehensive characterization of the tumour microenvironment,
immune cell infiltration, and cellular architecture in situ.

### Spatial transcriptomics

Spatial transcriptomics enables the study of gene expression within the
tissue architecture, preserving spatial context at cellular or
subcellular resolution. These platforms can be broadly divided into two
main categories based on detection strategy: imaging-based
methods---including [ISH]{acronym-label="ISH"
acronym-form="singular+short"} and [ISS]{acronym-label="ISS"
acronym-form="singular+short"} ---and spatial barcoding methods, which
rely on capture-based approaches followed by sequencing. Each approach
presents trade-offs in terms of resolution, transcriptome coverage,
throughput, and tissue
compatibility[@B35_liu_spatiotemporal2024; @B49asp2020; @B50_du2023; @B51_museumST_2022]
(Table [1.2](#tab:backg_st){reference-type="ref"
reference="tab:backg_st"}).

#### In Situ Hybridization Imaging-Based Approaches

ISH-based techniques use fluorescently labelled probes that hybridize
directly to target RNA molecules in fixed tissue sections. These methods
are highly accurate and can offer single-molecule and single-cell
resolution, but typically have limitations in multiplexing capacity
unless cyclic imaging or barcode strategies are used.

Basic [smFISH]{acronym-label="smFISH" acronym-form="singular+short"} can
detect individual transcripts with high spatial precision but is limited
in the number of genes detectable due to the finite number of
distinguishable fluorophores.

Advanced multiplexed methods, such as [MERFISH]{acronym-label="MERFISH"
acronym-form="singular+short"}, and seqFISH+, overcome these limits by
iterative cycles of hybridization and imaging, enabling detection of
hundreds to thousands of genes. MERFISH[@B52_chen2015] uses
combinatorial labelling and error-correcting barcodes to detect
thousands of RNA species in single cells.
[seqFISH]{acronym-label="seqFISH"
acronym-form="singular+short"}+[@B53_seqFISH+2019] leverages sequential
rounds of hybridization with an expanded pseudo-color palette, enabling
detection of over 10,000 genes at subcellular resolution.

Commercial platforms such as CosMx (Nanostring/Bruker
Spatial)[@B52_chen2015], MERscope (Vizgen), and Xenium (10x
Genomics)[@B56_G5_xenium] implement these strategies to achieve
high-resolution imaging of hundreds to thousands of RNA species, often
with optional co-detection of proteins.

#### In situ sequencing Imaging-Based Approaches

ISS techniques sequence RNA molecules directly within tissues,
preserving both spatial context and nucleotide identity. Unlike ISH, ISS
provides sequence information, enabling mutation and splice isoform
detection.

STARmap[@B57_STARmap2018] improves detection efficiency by using DNA
nanostructures and hydrogel-tissue chemistry, while
FISSEQ[@B58_FISSEQ2015], enables untargeted, whole-transcriptome
analysis through in situ reverse transcription and random-hexamer
priming. These methods retain spatial localization while offering a more
detailed molecular readout than hybridization alone.

Both ISH and ISS require fluorescence microscopy for imaging readouts
and are collectively referred to as imaging-based spatial
transcriptomics.

#### Spatial Barcoding and Sequencing-Based Methods

Unlike imaging methods, spatial barcoding techniques rely on
sequencing-based detection. They use spatially encoded oligonucleotides
(barcodes) fixed to a surface (e.g., slide, bead, or grid) to capture
RNA from overlying tissue. After RNA capture, reverse transcription and
sequencing are performed, and spatial information is reconstructed based
on barcode identity.

Prominent examples include Visium (10X Genomics)[@B59_stahl2016], which
captures RNA on slide-mounted barcoded spots ($\sim$`<!-- -->`{=html}55
µm resolution), and it's new version, Visium HD, that offers spatial
resolution to $\sim$`<!-- -->`{=html}2--5 µm. Other examples include
Slide-seq[@B60_slideseq2019] and Slide-seqV2[@B61_slideseqv2_2021] that
utilise barcoded beads with known spatial locations
($\sim$`<!-- -->`{=html}10 µm resolution).

These methods offer broader transcriptomic coverage---often approaching
the whole transcriptome---and are scalable to larger tissue sections.
However, they typically offer lower spatial resolution than
imaging-based platforms and may not reveal single-cell or subcellular
detail.

::: {#tab:backg_st}
  --------------- ----------------------------------------------- ------------------------------------ ------------------------------------ -------------
                                                                                                                                            
  **Coverage**                                                                                                                              
  **Scale**                                                                                                                                 
  smFISH          ISH                                                         1--10 genes               $\sim$`<!-- -->`{=html}200--300 nm  Subcellular
  MERFISH         ISH                                                         ,000+ genes               $\sim$`<!-- -->`{=html}100--300 nm  Subcellular
                                                                                                                                            
  seqFISH+        ISH-based                                                  \>10,000 genes               $\sim$`<!-- -->`{=html}200 nm     Subcellular
  CosMx           ISH / smFISH                                     $\sim$`<!-- -->`{=html}1,000 genes     $\sim$`<!-- -->`{=html}250 nm     Subcellular
  MERscope        MERFISH                                                                                                                   
  genes           $\sim$`<!-- -->`{=html}300 nm                               Subcellular                                                   
  Xenium          smFISH                                                                                                                    
  genes           $\sim$`<!-- -->`{=html}280 nm                               Subcellular                                                   
  STARmap         ISS                                                                                                                       
  genes           $\sim$`<!-- -->`{=html}2--3$\mu\text{m}$                    Single-cell                                                   
  FISSEQ          ISS                                                                                                                       
  transcriptome   $\sim$`<!-- -->`{=html}300 nm--1$\mu\text{m}$                                                                             
  Subcellular                                                                                                                               
  Visium          Spatial Barcoding (Slide)                                                                                                 
  transcriptome   $\sim$`<!-- -->`{=html}55$\mu\text{m}$                     Multicellular                                                  
  Visium HD       Spatial Barcoding (Slide)                                                                                                 
  transcriptome   $\sim$`<!-- -->`{=html}2--5$\mu\text{m}$                Close to single-cell                                              
  Slide-seq       Spatial Barcoding (Beads)                                                                                                 
  transcriptome   $\sim$`<!-- -->`{=html}10$\mu\text{m}$                        Cellular                                                    
  Slide-seqV2     Spatial Barcoding (Beads)                                                                                                 
  transcriptome   $\sim$`<!-- -->`{=html}10$\mu\text{m}$                        Cellular                                                    
  --------------- ----------------------------------------------- ------------------------------------ ------------------------------------ -------------

  : Comparison table of key features of spatial transcriptomics
  profiling methods.
:::

### Spatial metabolomics

Spatial metabolomics explores the spatial distribution of metabolites
directly in tissue sections, providing insight into biochemical activity
within the anatomical context[@B62_maldi2015]. The field is largely
driven by [MALDI-MSI]{acronym-label="MALDI-MSI"
acronym-form="singular+short"}[@B63_IMS2001], a label-free, untargeted
technique capable of detecting a wide range of small molecules including
lipids, neurotransmitters, and drugs.

In MALDI-MSI, tissue sections are coated with a matrix that facilitates
ionization when hit by a laser. The resulting ions are analysed by mass
spectrometry to reconstruct spatial metabolite maps. MALDI-MSI offers
spatial resolution in the range of 10--$50\,\mu\text{m}$, with coverage
of hundreds to thousands of metabolites, depending on the tissue and
matrix.

### Spatial multi-omics

Spatial multi-omics technologies
(Table [1.3](#tab:backg_smultiomics){reference-type="ref"
reference="tab:backg_smultiomics"}) enable the simultaneous or
integrative profiling of multiple molecular layers---such as RNA,
proteins, and metabolites---within their spatial tissue context,
offering a more comprehensive understanding of cellular states and
interactions.

One of the most established platforms, GeoMx [DSP]{acronym-label="DSP"
acronym-form="singular+short"}(NanoString)[@B64_merritt_multiplex_2020],
allows for high-plex profiling of both RNA and proteins within defined
regions of interest using oligonucleotide-tagged probes and UV-directed
barcode collection, though at limited spatial resolution
($\sim10–100\,\mu\text{m}$). Other advanced methods, such as
deterministic barcoding in tissue for spatial omics sequencing
(DBiT-seq)[@B65_liu2020], achieve co-detection of RNA and proteins
through microfluidic-based spatial barcoding on the same section,
offering high spatial resolution ($\sim 10\,\mu\text{m}$) and true
multimodal readouts. Similarly, Spatial-CITE-seq[@B66_CITESEQ_liu2023]
adapts [CITE]{acronym-label="CITE" acronym-form="singular+short"} to the
spatial dimension, enabling the capture of transcriptomes alongside
surface protein markers.

Furthermore, MALDI-MSI have been successfully combined on the same slide
with both IMC[@B67_nunes2024] and spatial transcriptomics platforms like
10x Visium[@B68_vicari2024].

::: {#tab:backg_smultiomics}
  ---------------------- ------------------------------------------- -------------------------------- ---------- --------------------------------------------
                                                                                                                 
  **Targets**                                                                                                    
  **Targets**                                                                                                    
  **Coverage**                                                                                                   
  **Resolution**                                                                                                 
  GeoMx DSP                     $\sim$`<!-- -->`{=html}18,000                      100+                   --     
  (ROI-based)                                                                                                    
  DBiT-seq                   $\sim$`<!-- -->`{=html}6,000-10,000      $\sim$`<!-- -->`{=html}30-100       --        $\sim$`<!-- -->`{=html}10$\mu\text{m}$
  Spatial-CITE-seq           $\sim$`<!-- -->`{=html}5,000-10,000      $\sim$`<!-- -->`{=html}100-200      --      $\sim$`<!-- -->`{=html}20--50$\mu\text{m}$
  MALDI-MSI + IMC                            --                        $\sim$`<!-- -->`{=html}30-50              
  1,000s                  $\sim$`<!-- -->`{=html}1--10$\mu\text{m}$                                              
  MALDI-MSI + Visium                        18000                                   --                 100-1000  
  55$\mu\text{m}$ (ST)                                                                                           
  ---------------------- ------------------------------------------- -------------------------------- ---------- --------------------------------------------

  : Comparison table of key features of spatial multi-omics methods.
:::

Altogether, spatial omics are accelerating insights into tissue biology
and transforming multiple areas of medicine. In particular, they have
been extensively applied to study complex microenvironments, such as
tumours, where spatial context and cellular heterogeneity are critical
and often obscured in bulk or single-modality
data[@B51_museumST_2022; @P24_abadi_tensorflow_2016; @B69_engblom2025; @B71_hsieh2022].
They have also been used to study clonality differences in
space[@B47_B72_S4_IMC2014; @B73_engblom_spatial_2023]. As the
applications are vast and rapidly evolving, they are reviewed in detail
elsewhere[@B51_museumST_2022; @B69_engblom2025; @B71_hsieh2022].

With this foundation in the underlying technologies and their biological
potential, the next section shifts focus to the analytical challenges
and computational frameworks required to interpret and integrate spatial
omics data.

## Analytical Frameworks for Spatial Omics Data

The increasing availability of high-resolution spatial omics
data---particularly at subcellular resolution---brings new computational
challenges, including those related to the spatial dimension, increased
data complexity, and multiplexing. This chapter focuses on the analysis
of single-molecule spatial transcriptomics and proteomics, where each
detected molecule is mapped with precise spatial coordinates. While the
field is rapidly evolving, with a growing number of specialized tools
and packages emerging at a fast pace, this work will centre on the core
analytical pipeline common to most imaging-based platforms. Methods
specific to certain technologies---such as deconvolution approaches
designed for lower-resolution platforms like 10x Visium---or complex
multimodal integration strategies with single-cell data will not be
covered here.

### Data preprocessing

Both spatial proteomics and spatial transcriptomics require dedicated
preprocessing steps to mitigate technical noise and artifacts that may
obscure true biological signals. These preprocessing strategies differ
significantly, reflecting the distinct nature of the underlying
technologies.

#### Spatial proteomics

For spatial proteomics noise sources can vary based on the antibodies,
detection channels, and tissue types used, and may manifest as artifacts
such as hot pixels, shot noise, background noise and channel
crosstalk[@P6_baharlou_mass_2019; @P7_baranski_maui_2021; @P8_chevrier_compensation_2018]
and may also exhibit low signal-to-noise
ratios[@P9_12_milosevic_different_2023].

Channel crosstalk, where signals from one channel interfere with
adjacent channels, can be mitigated through well-designed antibody
panels[@P9_12_milosevic_different_2023] or correction methods like
CATALYST[@P8_chevrier_compensation_2018], which uses pre-acquisition
compensation matrices, as well as post-processing
techniques[@P7_baranski_maui_2021; @P11_wang_multiplexed_2019].

Beyond crosstalk, other sources of noise---such as hot pixels,
background signal, and shot noise---require distinct computational
strategies. Some studies have employed adjacent approaches, including
those implemented in Steinbock[@B79_P15_S20_steinbock2023] and MAUI (MBI
analysis interface)[@P7_baranski_maui_2021], while others have developed
"homebrew" methods based on traditional image filtering
techniques[@P11_wang_multiplexed_2019; @P13_keren_structured_2018; @B81_P14_rendeiro2021].
However, these methods are often insufficient for fully addressing
complex noise patterns.

Tools like Ilastik[@B82_P16_ilastik2019] provide supervised pixel
classification to distinguish background noise from true signal on a
per-marker
basis[@B79_P15_S20_steinbock2023; @B82_P16_ilastik2019; @B83_P17_25_ijsselsteijn2021].
Although this approach requires extensive manual annotation, it is
currently considered state-of-the-art, as it effectively removes
background noise and improves signal normalization and batch effect
reduction across samples[@P13_keren_structured_2018]. More recently,
IMC-Denoise[@P10_lu_imc-denoise_2023] has been introduced as a two-step
pipeline combining traditional algorithms with a self-supervised deep
learning model based on Noise2Void[@P19_krull_noise2void_2019].

Notably, this thesis contributes to this specific challenge, as
discussed in Chapter [\[ch:penguin\]](#ch:penguin){reference-type="ref"
reference="ch:penguin"}.

#### Spatial transcriptomics

For spatial transcriptomics, preprocessing aims to reduce technical
noise introduced during tissue handling, permeabilization, probe
hybridization, and sequencing. One of the first steps involves removing
low-quality spatial barcodes or transcript reads, which can arise from
ambient RNA contamination, incomplete permeabilization, or sequencing
artifacts. In some commercial platforms (such as Xenium), quality scores
or confidence values are assigned to individual transcripts; Transcripts
below a defined threshold can be excluded to avoid false positives. To
further refine the dataset, transcripts detected at extremely low
frequency across the tissue may be removed.

### Cell segmentation

A central challenge in spatial omics is the accurate assignment of
molecular measurements---transcripts or proteins---to individual cells.
This process, known as cell segmentation, is essential for translating
spatial data into single-cell level insights and directly influences
downstream biological interpretation.

#### Spatial proteomics

In spatial proteomics, cell segmentation is typically performed using
the imaging data generated during acquisition. These multiplexed images
often include nuclear stains to identify nuclei and membrane markers to
delineate cell boundaries. These signals are combined into RGB or
multi-channel images and processed using image analysis pipelines.

Traditional image segmentation approaches, such as thresholding, edge
detection, and the watershed algorithm, are commonly employed. These
methods are implemented in established tools like
CellProfiler[@P26_S21_mcquin_cellprofiler_2018; @S7_cellprofiler2006]
and ImageJ/Fiji[@B88_schindelin_fiji_2012], which allow customization of
segmentation pipelines using scripting interfaces. In recent years, DL
models have gained prominence due to their improved accuracy,
generalizability, and community availability[@B89_zerocostDL4MIC].
Notably, Mesmer, part of the DeepCell framework, DL-based model
specifically trained on spatial proteomics datasets, leveraging
TissueNet, a large annotated dataset[@S5_mesmer2022]. Another widely
adopted method is Cellpose, a generalist DL segmentation tool that
predicts vector flows to delineate cell boundaries instead of relying on
direct pixel-wise classification. Though originally trained on a broad
array of microscopy images, Cellpose also offers specialist models,
including those fine-tuned on TissueNet subsets to enhance performance
on spatial proteomics images[@S6_cellpose2021; @S16_cellpose2].

Segmentation strategies in spatial proteomics range from nucleus-based
expansion methods, which approximate cell outlines from DAPI staining,
to full-cell segmentation techniques that incorporate both nuclear and
membrane markers. A growing number of domain-specific models continue to
emerge, each tailored to address the distinct noise profiles, spatial
resolution, and multiplexing characteristics of spatial proteomics
data[@B93_lee_cellseg_2022; @B94_mcmicro2022; @B95_matisse2021; @B96_splinedist2020; @B97_stardist2018].

This thesis also contributes to this domain, as discussed in
Chapter [\[ch:Cellcytolego\]](#ch:Cellcytolego){reference-type="ref"
reference="ch:Cellcytolego"}.

#### Spatial transcriptomics

The landscape of spatial transcriptomics segmentation is more varied
than in proteomics. In subcellular-resolution platforms such as Xenium
and CosMx, segmentation is often performed using accompanying IF images,
which typically include nuclear (e.g., DAPI) and membrane markers. These
images enable the identification of cell boundaries, which are then used
to assign transcripts to individual cells by overlaying segmentation
masks with transcript positions.

Alternatively, several methods bypass image-based segmentation
altogether. Since transcripts are detected directly as spatial
coordinates, these approaches aim to infer cell boundaries based solely
on the spatial distribution and density of transcripts, reducing
dependency on auxiliary
images[@G8_baysor; @G9_ssam; @B100_prabhakaran_sparcle_2022; @B101_clustermap_2021; @B102_ficture_2024; @B103_bidcell_2024; @B104_defard2024; @B105_spage2vec2021; @B106_qian_probabilistic_2020].

A prominent example is Baysor[@G8_baysor], a general probabilistic
framework implemented in Julia. Based on [MRF]{acronym-label="MRF"
acronym-form="singular+short"}, Baysor models cells by jointly
considering spatial proximity and transcriptional similarity. It can
incorporate prior segmentations, assigning confidence scores to them, or
operate de novo using only transcript data. Baysor functions in both 2D
and 3D, optimizing the likelihood that neighbouring transcripts
originate from the same cell. Notably, it has shown strong performance
across datasets and has been reported to outperform the default
segmentation provided in Xenium workflows[@B107_optXenium2025],
highlighting its practical utility and growing adoption.

### Cell phenotyping

Cell phenotyping is the process of assigning biological identities to
individual cells based on their molecular profiles, such as gene or
protein expression, and serves as a critical step in interpreting
spatial omics data.

#### Spatial proteomics

Unlike transcriptomic approaches, spatial proteomics typically profiles
a pre-selected panel of $\sim$`<!-- -->`{=html}30--50 protein markers,
chosen based on the biological question. This reduced and curated
feature space simplifies downstream computational analysis but increases
reliance on prior biological knowledge for interpreting
clusters[@B79_P15_S20_steinbock2023].

Commonly used clustering algorithms include Cytosplore, a GUI which
leverages [HSNE]{acronym-label="HSNE"
acronym-form="singular+short"}[@S17_imacyte], Phenograph that builds a
shared nearest neighbor graph and uses Louvain algorithm to partition of
the graph into communities[@S17_imacyte], and FlowSOM that applies
[SOM]{acronym-label="SOM"
acronym-form="singular+short"}[@S19_flowsom2015; @B111_flowSOMprotocol2021].

In addition to unsupervised clustering, other strategies for phenotyping
include: Manual gating, where marker thresholds are used to define known
populations based on expert knowledge; Supervised machine learning
models, such as Random Forests, which classify cells using labelled
training data and reference mapping, where new datasets are aligned to
previously annotated references to infer phenotypes.

#### Spatial transcriptomics

Spatial transcriptomics datasets that approach single-cell resolution
often adopt analysis pipelines developed for
[scRNA]{acronym-label="scRNA" acronym-form="singular+short"}. This
involves several key steps:

The process begins with data preprocessing, where cells of low quality
are removed through quality control steps. Following this, normalization
is applied---most commonly scaling gene counts to 10,000 per cell and
applying a log-transformation---to standardize data across cells. Since
these datasets are high-dimensional, dimensionality reduction is used to
make the data more manageable. [PCA]{acronym-label="PCA"
acronym-form="singular+short"} is commonly applied first to identify the
most informative gene expression patterns. For visualization and further
analysis, nonlinear techniques like t-SNE and UMAP are used to project
the data into two or three dimensions while preserving its underlying
structure[@B112_bestPractices2023].

Next, clustering algorithms group similar cells together. This is
usually done by building a [KNN]{acronym-label="KNN"
acronym-form="singular+short"} graph in the PCA-reduced space. Each cell
is connected to its K most similar neighbours, and clustering algorithms
such as Louvain[@B113_blondel2008] and Leiden[@B114_waltman2013], often
with better performance[@B115_louvain_leiden2019] are applied to
identify densely connected communities within the graph. These clusters
are assumed to represent groups of cells with similar transcriptional
profiles, often corresponding to distinct cell types or states.

Once clusters are identified, cell type annotation is performed to
assign biological identities to the groups of cells. This can be done
either manually or automatically.

Manual annotation relies on known marker genes---genes that are
characteristically expressed in specific cell types. In practice, this
can be done by checking whether known markers are expressed in each
cluster or, conversely, by identifying differentially expressed genes
within clusters and matching them to known biological signatures. While
manual annotation is transparent and interpretable, it can be
subjective, labour-intensive, and limited by the availability and
specificity of known markers[@B112_bestPractices2023].

Alternatively, automated annotation methods are also available, which
leverage reference gene atlases and machine/deep learning models (e.g.,
CellTypist[@B116_dominguez2022]), or label transfer approaches (e.g.,
Symphony[@B117_symphony]). These methods provide faster and more
scalable annotation but tend to be less interpretable and dependent on
the similarity between the reference and the query
data[@B112_bestPractices2023].

Currently, automated approaches are not yet standard for spatial
transcriptomics, particularly in cases where the data is limited by
lower gene counts, targeted panels, or lower sequencing depth compared
to single-cell RNA. Additionally, many spatial datasets do not align
well with existing reference atlases due to differences in tissue
context, resolution, or platform. As a result, manual annotation remains
more reliable and commonly used in spatial transcriptomics, particularly
when high-resolution spatial context and known marker expression are
available.

### Cellular neighbourhood analysis

Cellular neighbourhood analysis begins with the construction of a
spatial connectivity graph that captures the proximity relationships
between cells within the tissue. This spatial graph serves as the
backbone for downstream spatial statistics and interaction analyses.
Different methods can be used to construct such graphs, including
k-nearest neighbours, where each cell is connected to a fixed number of
closest neighbours; radius-based expansion, which connects all cells
within a given physical distance; and Delaunay triangulation, which uses
geometric criteria to define adjacency without imposing arbitrary
thresholds. Importantly, the same principles and methodologies for
neighbourhood analysis apply across these two modalities, despite
differences in the underlying molecular
readouts[@B79_P15_S20_steinbock2023; @B112_bestPractices2023].

Once a graph is established, it enables the exploration of spatial
interactions between annotated cell types. A common approach involves
the computation of neighbourhood enrichment scores, which statistically
test whether specific cell types are found adjacent to each other more
or less frequently than expected by chance. These analyses rely on
permutation-based null models, which shuffle cell-type labels while
preserving tissue structure, providing a robust statistical framework to
assess enrichment or depletion of interactions[@B118_histocat_2017].
Another related method is the computation of co-occurrence scores, which
estimate how likely it is to observe specific cell-type pairs within
increasing radii around each cell. These scores reflect the conditional
probability of observing a certain cell type given the presence of
another nearby, offering an interpretable measure of spatial association
across scales[@B112_bestPractices2023].

Alternatively, simple interaction matrices can be computed, summarizing
the raw counts of neighbouring cell-type
pairs[@B79_P15_S20_steinbock2023]. Though not statistical tests, these
matrices are useful for exploratory data analysis.

Beyond pairwise interaction analyses, more integrative approaches aim to
cluster cells based on the composition of their local neighbourhoods.
One strategy involves computing, for each cell, the fraction of
surrounding cell types within its local environment (e.g., 20 nearest
neighbours). These local composition profiles can then be clustered
using unsupervised methods such as k-means or Leiden algorithms,
grouping cells into recurring spatial neighbourhoods. This approach was
introduced by studies such as Goltsev et al.[@P5_B44_S2_CODEX2018] and
Schürch et al.[@B119_schurch_coordinated_2020], which revealed that
specific combinations of neighbouring discrete cell types are often
spatially organized in conserved patterns.

A related method involves aggregating gene or protein expression
features across the neighbourhood, effectively capturing a summary of
the local microenvironment. These aggregated features can then serve as
the basis for clustering or downstream modelling, this is approach is
implemented for example in covariance environment
(COVET)[@B120_covet2025], NicheCompass, a graph deep-learning approach
to identify and quantitatively characterize niches by learning cell
embeddings encoding signalling events as spatial gene program
activities[@B121_birk_quantitative_2025] and Banksy[@B_banksy_2024].

Through these computational strategies, spatial neighbourhoods can be
identified and interpreted biologically as spatial niches---localized,
recurring configurations of cells and their molecular environment that
reflect functionally relevant tissue microenvironments. Spatial niches
are particularly informative in contexts such as immuno-oncology, where
immune-tumour interactions shape the progression or suppression of
disease.

### Spatial domains

Closely related to cellular neighbourhoods, spatial domains usually
refer to broader, tissue-scale regions characterized by coherent gene or
protein expression patterns and underlying structural
organization[@B112_bestPractices2023]. While cellular neighbourhoods
capture the immediate microenvironment around a cell---defined by local
interactions and spatial proximity---spatial domains extend this concept
to larger anatomical or functional areas of tissue. These regions often
consist of diverse cell types and multiple neighbourhood configurations,
working together to support complex biological functions.

Because of the overlap between microenvironmental patterns and larger
spatial structures, some computational approaches can identify both
neighbourhoods and domains using similar principles. These methods
generally integrate molecular profiles with spatial information,
allowing for the detection of both fine-grained and broader-scale tissue
organization. A common strategy involves combining a cell's individual
gene or protein expression with the aggregated expression from its
surrounding neighbours. This allows the model to capture both intrinsic
cell identity and the influence of the local microenvironment, which is
particularly useful in tissues where cells of the same type are
dispersed rather than spatially clustered.

A wide range of computational strategies has been developed for domain
segmentation. Some early methods relied on spatial smoothing techniques,
such as Markov Random Fields, which encourage physically proximate cells
to be assigned the same label[@B122_bayespace2021; @B123_zhu2018]. This
assumes that a cell's transcriptome resembles the average of its domain,
although this may not always hold true in tissues where multiple cell
types are intermixed. Other approaches employ deep learning,
particularly graph-based neural networks that can incorporate either
histological information[@B124_pham2023; @G10_b125_spagcn] or spatial
graphs derived purely from molecular and positional
data[@B126_dong2022; @B127_graphst2023; @G14_P2R]. These methods offer
flexibility and can capture complex spatial dependencies, but their
performance can vary with dataset size and structure.

More recently, methods introducing hierarchical and multiscale
representations of spatial structure have emerged. For instance,
NeST[@b129_nest2023] identifies nested co-expression
hotspots---spatially contiguous regions that co-express subsets of
genes---by simultaneously searching across gene and spatial dimensions.
This approach captures biological structure at multiple scales and can
reveal overlapping or nested domains, without requiring prior
assumptions about gene sets or spatial resolution. Similarly, the
concept of tissue schematics[@b130_tissueschem2022] has been proposed as
a way to abstract spatial organization into higher-order motifs, by
identifying modular assemblies of cellular neighbourhoods. These
schematics can be used to represent tissue architecture in both healthy
and diseased states, offering insights into how local interactions scale
up to tissue-level functionality.

### Spatially Variable genes

[SVG]{acronym-label="SVG" acronym-form="singular+short"}s are genes
whose expression displays significant spatial patterning across a
tissue. Identifying SVGs reveals insights into tissue architecture,
cell--cell communication, and local microenvironmental influences,
extending beyond what is captured by cell types or spatial domains
alone. These spatial patterns may arise from gradients in signalling
molecules, regional functional differences, or heterogeneous cell
compositions[@B112_bestPractices2023]. Notably, SVG detection does not
rely on cell segmentation, making it applicable across different spatial
omics technologies.

Several computational methods have been developed to detect SVGs by
decomposing gene expression variability into spatial and non-spatial
components. A widely used statistic is Moran's I, which quantifies
spatial autocorrelation by measuring how similar gene expression is
between neighbouring spots[@B131_getis2010]. Model-based approaches like
SpatialDE[@B132_G12_spatialDE2018], trendseek[@B133_edsgard2018],
SPARK[@B134_sun2020], and SPARK-X[@B135_sparkX2021] use Gaussian
processes or spatial statistical models to test for spatially structured
expression while accounting for noise and overdispersion. While these
tools test each gene independently and return p-values, they often
overlook spatial domain context, which can limit biological
interpretability. Other strategies take different modelling
perspectives: Sepal[@B136_sepal_2021] applies a Gaussian diffusion
process, scGCO[@B137_scGCO2022] uses graph cuts to detect spatial
expression boundaries, and SpaGCN[@G10_b125_spagcn], defines both
domains and SVGs integrating histology and spatial proximity through
graph convolutional networks.

Importantly, the identification of SVGs is modality-agnostic and
applicable to both spatial transcriptomics and spatial proteomics.
Regardless of whether gene expression or protein abundance is measured,
spatially variable features represent key molecular signatures of tissue
structure and function, and form an essential layer of spatial omics
analysis.

### Functional analysis

Understanding how spatial context shapes cellular function is an
increasingly important focus in spatial omics. Beyond identifying cell
types, neighbourhoods, or spatially variable genes, functional analysis
aims to reveal how cells operate and interact in situ. This includes
investigating intracellular signalling pathways, [TF]{acronym-label="TF"
acronym-form="singular+short"} activity, and particularly
c[CCC]{acronym-label="CCC" acronym-form="singular+short"} events---often
inferred from curated interaction networks derived from transcriptomics
data[@B138_armingol2024; @B139_liana2024].

A broad landscape of computational tools has emerged for CCC
analysis[@B139_liana2024; @B140_fischer2023; @B141_cellphonedb_2025; @B142_cellchat2025; @B143_spatalk2022; @B144_opttransp2023; @B145_spatialDM2023; @B146_nichenet2020; @B147_gcng2020; @B148_zhu_mapping_2024],
reflecting the growing complexity and richness of spatial omics data.
Broadly, these methods fall into two sections: identifying pairs of
genes that interact, such that expression of the gene in one cell
influences that of the other gene in others; and identifying pairs of
cells in which that gene pair interacts[@B149_walker2022].

Most methods for CCC leverage [L-R]{acronym-label="L-R"
acronym-form="singular+short"} interactions, relying heavily on prior
biological knowledge, often curated into databases such as
CellPhoneDB[@B141_cellphonedb_2025], OmniPath[@B150_omnipath2016], and
CellChat[@B142_cellchat2025]. These databases catalogue known signalling
pathways and interaction pairs from the literature. Using such
resources, CCC inference can be extended to the spatial context by
identifying L--R pairs co-expressed in nearby cells[@B139_liana2024] for
example using spatial
co-expression[@B139_liana2024; @B145_spatialDM2023], or incorporating
different approaches such as optimal transport
frameworks[@B144_opttransp2023] or Bayesian [MIL]{acronym-label="MIL"
acronym-form="singular+short"}[@B148_zhu_mapping_2024] to more precisely
link cells by both location and expression.

Complementing these prior knowledge-driven approaches are data-driven
methods that model spatial gene expression to discover novel
interactions. These models, like NCEM[@B140_fischer2023],
GCNG[@B147_gcng2020] or SVCA[@B151_svca2019] and Misty[@B152_misty2022]
(while not specific for CCC), aim to capture interactions that explain
spatial expression variance across multiple genes. They enable inference
of new communication patterns not captured in curated L--R databases,
though their accuracy depends on model design and training data.

Furthermore, some CCC frameworks estimate global co-localization across
entire tissue
sections[@B140_fischer2023; @B143_spatalk2022; @B152_misty2022], and
others focus on local cell-cell
proximity[@B144_opttransp2023; @B145_spatialDM2023].

In parallel, there is increasing interest in inferring intracellular
functional activity---such as pathway activation, TF activity, or gene
set enrichment---within individual cells or regions. Tools like
decouplR[@B153_decoupler2022] apply prior knowledge from pathway
perturbation signatures[@B154_progeny2018] or regulatory
networks[@B155_collectri2023] to estimate these functional states at
single-cell or spot level. As with CCC, these functional scores can then
be spatially mapped. These approaches are currently more mature for
spatial transcriptomics, where broader gene coverage enables more
reliable functional inference. In contrast, spatial proteomics remains
limited by smaller marker panels and less comprehensive prior knowledge,
constraining the resolution of functional state estimation.

While spatial omics provides a key advantage over single-cell RNA
through direct measurement of cell proximity---eliminating the need for
probabilistic neighbourhood modelling and enabling exclusion of
implausible, long-range interactions---this often comes with trade-offs.
Platforms with true cellular resolution (e.g., CosMx, Xenium) may
capture fewer genes, complicating pathway-level analyses and downstream
functional interpretation. Moreover, the heavy reliance on curated
databases across most CCC methods introduces variability and potential
inconsistencies, as predictions can differ significantly depending on
the resource used.

Ultimately, understanding how a cell's functional state is shaped by its
environment---and how it, in turn, influences surrounding cells---is a
central and still largely untapped frontier in spatial biology. As
methods advance, this bidirectional view of cellular function in context
promises to transform our understanding of tissue organization, disease
progression, and therapeutic targeting.

### How to distinguish between sample conditions

While the tasks described in the previous sections---such as
preprocessing, cell segmentation, and feature extraction---lay the
foundation for spatial omics analysis, deriving meaningful biological
insights often requires comparing groups, such as different disease
states, treatment responses, or tissue types.

#### Summary statistics

One of the earliest approaches involves aggregating spatial features
across samples and comparing them using statistical summaries. For
example, scores for cell--cell communication, number of neighbours, or
other spatial statistics can be averaged per sample and used to detect
differences between
conditions[@B139_liana2024; @B152_misty2022; @G7_squidpy]. However, this
strategy is often too simplistic, as it can blur important heterogeneity
by averaging out spatially localized effects.

#### Matrix factorization methods

Matrix factorization methods provide a more nuanced alternative. These
approaches reduce high-dimensional data into a smaller set of latent
factors that capture major patterns of variation. Classical approaches
like PCA achieve this through linear decompositions, but newer tools
like [MOFA]{acronym-label="MOFA"
acronym-form="singular+short"}[@B157_MOFA2018; @B158_mofa+2020; @B159_MOFAcell2023],
MEFISTO[@B160_mefisto2022], and DIALOGUE[@B161_dialogue2022] extend the
concept to handle multiple data types, structured variation, or
intercellular coordination. These methods can incorporate spatial
information embedded in the matrix.

MOFA and its variants allow joint analysis of multi-omic datasets,
identifying latent factors that capture both shared and group-specific
variation. Other tools like DIALOGUE[@B161_dialogue2022] and
scITD[@B162_tensor2024] focus on identifying coordinated gene programs
across multiple cell types, enabling the characterization of
multicellular processes in an unsupervised fashion.

[NMF]{acronym-label="NMF" acronym-form="singular+short"} provides a more
interpretable, parts-based representation, especially useful for
identifying additive biological signals like distinct cell states or
spatial domains. Spatially-aware versions of NMF, such as
[NSF]{acronym-label="NSF" acronym-form="singular+short"}, extend this
idea by directly incorporating spatial coordinates, better capturing
localized gene expression patterns[@B163_NNMF2023].

These matrix-based methods not only enhance the ability to uncover
biologically relevant spatial and intercellular patterns, but also
enable unsupervised identification of group-level differences---whether
driven by disease, environment, or treatment---without requiring
explicit labels.

#### Machine Learning and Deep Learning approaches

Beyond matrix factorization, supervised [ML]{acronym-label="ML"
acronym-form="singular+short"} and [DL]{acronym-label="DL"
acronym-form="singular+short"} methods are increasingly being explored
to classify samples or predict disease conditions using spatial and
single-cell omics data. These models are particularly well-suited to
capturing nonlinear relationships and subtle patterns in
high-dimensional datasets. However, their application in this domain
remains relatively limited---mainly due to the scarcity of large,
well-annotated datasets needed to train robust deep models. While DL
approaches typically require thousands of samples to achieve
generalizability, spatial omics datasets have only recently begun to
reach that scale.

It's worth noting that DL has been widely adopted in other stages of the
spatial omics pipeline---such as image preprocessing, segmentation,
domain identification, phenotyping, and modelling of cell--cell
communication---but its use for direct classification or outcome
prediction is still sparse.

Among early efforts in this direction, NaroNet[@B164_naronet2022]
introduced a patch contrastive learning strategy combined with graph
representations to predict patient outcomes from IMC data. In the same
cohort, Fu et al.[@B165_fu2023] proposed a deep multimodal graph-based
network that integrates IMC data with clinical variables to predict
cancer survival. In another example, Risom et al.[@B166_risom2022]
applied a random forest classifier to a MIBI dataset of ductal carcinoma
in situ (DCIS), using over 400 spatial features---including tissue
compartment enrichment and TME morphometrics---to distinguish between
progressors and non-progressors.

A more recent and notable advance is S3-CIMA[@B167_s3cima2023], a weakly
supervised, single-layer convolutional neural network that learns
disease-specific spatial compositions of the tumour microenvironment. By
modelling local cell-type organizations from high-dimensional proteomic
imaging data (IMC and CODEX), it enables the discovery of
outcome-associated microenvironments in colorectal cancer. Similarly,
graph deep learning has been used to predict prognosis in gastric
cancer, where Cell-Graphs built from multiplexed immunohistochemistry
(mIHC) data enable prognosis prediction from spatial arrangements of
cell types[@B168_wang2022].

Important to note is that DL methods are already well-established in
computational pathology and digital histopathology, where large
annotated datasets and well-defined visual features have allowed CNNs to
thrive in image classification, segmentation, and prognosis prediction
tasks[@B169_perezGuide2024; @B170_review2024].

### Frameworks and tools for spatial omics analysis

A growing ecosystem of software tools supports the analysis of
single-cell and spatial omics data. Seurat (R)[@B171_hao2021] and Scanpy
(Python)[@G6_scanpy_2018] are widely used for single-cell analysis.
Scanpy is built around the efficient AnnData structure[@g25_anndata],
while Seurat uses its own SeuratObject. Dedicated spatial omics tools
such as Giotto (R)[@B174_giotto2021] and Squidpy (Python)[@G7_squidpy]
offer integrated workflows for spatial statistics, neighbourhood
analysis, and visualization with also new data frameworks like
SpatialData (Python)[@B175_spatialdata2025] emerging to better support
spatial modalities. Napari[@g17_napari], a general-purpose image viewer,
complements these tools with interactive, high-dimensional image
visualization---useful for working with spatial coordinates and tissue
images.

Python and R remain the dominant programming languages in this space.
Python is increasingly favoured for spatial omics due to its speed,
memory efficiency, and compatibility with deep learning and image
processing (e.g., PyTorch), while R remains popular for exploratory
analysis and visualization[@B51_museumST_2022]. Meanwhile, Julia is
gaining attention for its performance, with tools like
Baysor[@G8_baysor] highlighting its potential in spatial omics
analysis[@B177_julia2023].

Given the rapid growth of tools---and the risk of incompatibilities in
data formats, APIs, and user interfaces---standardization efforts like
scverse[@B178_scverse2023] have emerged. These initiatives promote
well-maintained, interoperable core functionality, supporting a more
cohesive and collaborative spatial omics software ecosystem.

### Closing remarks

This review has focused on spatial omics methods that operate on the
basis of cell-defined units. However, it is important to note that many
of these approaches can also be applied at the transcript level,
bypassing the need for explicit cell segmentation---this is particularly
true for tasks such as domain identification. While cell-based analysis
remains the prevailing standard in spatial omics, alternative strategies
that complement or replace cell segmentation are gaining
ground[@G14_P2R; @G15_pixie]. This shift is partly driven by the
technical difficulty of accurately segmenting individual cells,
especially in complex tissues where cells may be overlapping, densely
packed, or poorly defined. Acknowledging these challenges, this thesis
introduces novel approaches for cell-free analysis of spatial
transcriptomics data, as elaborated in
Chapter [\[ch:gridgen\]](#ch:gridgen){reference-type="ref"
reference="ch:gridgen"}.

Furthermore, this review has specifically addressed computational
methods applicable to single-cell resolution spatial data, and does not
cover broader-scale approaches such as spatial deconvolution or data
imputation, which are more relevant to spot-based or lower-resolution
platforms.

Even within the focused scope of single-cell approaches, many valuable
tools and developments could not be fully covered, reflecting the sheer
volume and velocity of innovation in the field.
Figure [1.1](#fig:review){reference-type="ref" reference="fig:review"}
provides a visual overview of the spatial omics methods discussed in
this review, highlighting the diversity and complexity of the
computational landscape. Nodes are organized by their functional role in
the analysis pipeline and scaled by citation count, illustrating both
methodological clustering and community impact. Furthermore, an
interactive visualization is available at
<https://marta-seq.github.io/SOME/>. The increasing availability of
spatially resolved data across multiple modalities has fuelled an
avalanche of computational methods, making it increasingly difficult for
researchers to keep pace and make informed decisions about which tools
best suit their needs. In this evolving landscape, there is a growing
need for \"living reviews\"[@B51_museumST_2022], curated repositories,
and community-driven benchmarking[@B170_review2024] efforts that can
adapt to the field's rapid progress.

![Graph overview of spatial omics methods analysed in this review. Each
node represents a method included in this review, with the colour
indicating its role in the spatial omics analysis pipeline (e.g., orange
-- Preprocessing, blue -- Cell Segmentation, etc.) and the size of the
bubble proportional to its citation count, reflecting community
adoption. Nodes are grouped spatially by conceptual similarity and
method type. Only full spatial omics methods are shown, excluding
foundational algorithms or reused modules from unrelated pipelines. This
layout reveals the landscape and complexity of spatial omics
development. The interactive version of this graph is available at:
<https://marta-seq.github.io/SOME/>. This approach can be extended to
incorporate newly published methods or customized for other use
cases.](Chapters/background/review.pdf){#fig:review width="\\textwidth"}
