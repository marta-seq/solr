---
bibliography: bibliography.bib
csl: apa.csl
---

Understanding the complexity of cancer, particularly its cellular
heterogeneity and dynamic microenvironment, relies heavily on advanced
analytical technologies.

Traditionally, cancer research has been dominated by genomic and
transcriptomic approaches, which have illuminated key mutations,
epigenetic alterations, and dysregulated pathways across tumour types
(Hoadley et al., 2018; L. Nunes et al., 2024; The Cancer Genome Atlas
Network, 2012; The Cancer Genome Atlas Research Network et al., 2013).
More recently, multi-omics strategies---including epigenomics (Baylin &
Jones, 2011; X. Chen et al., 2025; Li, 2021), proteomics (Aebersold &
Mann, 2016; Suhre et al., 2021; Uhlén et al., 2015), and metabolomics
(Schmidt et al., 2021) --- have expanded our understanding of tumour
biology by capturing additional layers of molecular regulation.

Single-cell technologies have further revolutionized the field by
enabling the dissection of intra-tumoral heterogeneity at cellular
resolution, uncovering rare cell populations and lineage trajectories
that bulk analyses obscure (Aldridge & Teichmann, 2020; Kinker et al.,
2020). These innovations have also extended to single-cell proteomics
(Mansuri et al., 2023; Setayesh et al., 2023; "Single-Cell Proteomics,"
2023) and metabolomics (Hu et al., 2023; Seydel, 2021), offering
increasingly comprehensive insights into tumour ecosystems. However,
despite their power, these methods often lack spatial context---an
essential component for understanding how cellular localization and
cell--cell interactions within the tumour microenvironment influence
disease progression and treatment response.

On the other hand, spatial information is routinely obtained in the
clinic through histopathological staining (e.g., H&E),
[IHC]{acronym-label="IHC" acronym-form="singular+short"}, and
radiological imaging (Clark et al., 2013). These approaches offer
spatial resolution at the tissue or organ level and remain vital for
diagnosis and clinical decision-making, but are inherently limited in
molecular detail and throughput, lacking the capacity to resolve
cellular heterogeneity or dynamic molecular interactions at scale.

To bridge this gap, spatial omics technologies have emerged as powerful
tools that integrate high-resolution molecular profiling with spatial
localization. These approaches allow the mapping of transcripts,
proteins, and metabolites directly within their tissue
context---preserving the native architecture of tumours and their
microenvironments.

# Spatially Resolved Omics Techniques

Spatial omics technologies have the potential to advance our
understanding of tumour ecosystems and improve clinical outcomes. The
essence of spatial omics lies in its aptitude for the simultaneous
detection of molecular constituents at exact spatial coordinates (L. Liu
et al., 2024). However different techniques vary greatly in resolution,
scale and molecular complexity (Palla et al., 2022).

## Spatial proteomics

Spatial proteomics encompasses technologies that enable the in-situ
profiling of proteins within tissues, preserving their spatial
localization. Most spatial proteomics techniques detect proteins using
antibodies tagged with fluorophores, metals, or DNA barcodes. These tags
are then read using technologies such as fluorescence microscopy, mass
spectrometry, or DNA-based imaging to map the spatial distribution of
proteins within tissues (Hickey et al., 2022; Jing et al., 2025; L. Liu
et al., 2024; Lundberg & Borner, 2019). A summary of key features across
these technologies is provided in
Table [1](#tab:spatial_proteomics){reference-type="ref"
reference="tab:spatial_proteomics"}.

Immunohistochemistry (IHC) is one of the most established clinical tools
for protein detection (Magaki et al., 2019). It uses enzyme-linked
antibodies to generate a chromogenic signal visible under brightfield
microscopy. While widely available and routinely used in diagnostics,
traditional IHC is generally limited to detecting one or a few markers
per tissue section, making it unsuitable for high-dimensional spatial
profiling.

Among the fluorescence-based approaches, [IF]{acronym-label="IF"
acronym-form="singular+short"}(Ijsselsteijn et al., 2019) is widely used
but limited in multiplexing. More advanced cyclic immunofluorescence
methods---such as [t-CyCIF]{acronym-label="t-CyCIF"
acronym-form="singular+short"}(Lin et al., 2018) and
[IBEX]{acronym-label="IBEX" acronym-form="singular+short"}(Radtke et
al., 2022) - use iterative staining and imaging cycles to overcome
spectral limitations, enabling the detection of 40--60+ markers with
spatial resolution down to 200--300 nm.

DNA-barcoded approaches such as [CODEX]{acronym-label="CODEX"
acronym-form="singular+short"} (Goltsev et al., 2018) and
[Immuno-SABER]{acronym-label="Immuno-SABER"
acronym-form="singular+short"}(Saka et al., 2019) further enhance
multiplexing. CODEX utilizes DNA-barcoded antibodies and sequential
hybridization of fluorescent probes, achieving high multiplexing (up to
60 proteins) with single-cell resolution ($\sim$`<!-- -->`{=html}500 nm)
in a single imaging plane (Black et al., 2021). Immuno-SABER employs
orthogonal DNA concatemers for signal amplification achieving 32-plex
with same resolution as COSMX (Saka et al., 2019).

Mass spectrometry-based approaches---notably [IMC]{acronym-label="IMC"
acronym-form="singular+short"} (Giesen et al., 2014) and
[MIBI-TOF]{acronym-label="MIBI-TOF" acronym-form="singular+short"}(Keren
et al., 2019)---use antibodies conjugated to isotopically pure
lanthanide metals, which are detected using laser ablation (IMC) or ion
beams (MIBI-TOF). These methods avoid fluorescence background and allow
simultaneous quantification of 40--50 proteins per tissue section. IMC
offers spatial resolution of approximately 1 μm, while MIBI achieves
higher resolution ($\sim$`<!-- -->`{=html}300 nm), albeit with more
complex instrumentation.

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

## Spatial transcriptomics

Spatial transcriptomics enables the study of gene expression within the
tissue architecture, preserving spatial context at cellular or
subcellular resolution. These platforms can be broadly divided into two
main categories based on detection strategy: imaging-based
methods---including [ISH]{acronym-label="ISH"
acronym-form="singular+short"} and [ISS]{acronym-label="ISS"
acronym-form="singular+short"} ---and spatial barcoding methods, which
rely on capture-based approaches followed by sequencing. Each approach
presents trade-offs in terms of resolution, transcriptome coverage,
throughput, and tissue compatibility (Asp et al., 2020; Du et al., 2023;
L. Liu et al., 2024; Moses & Pachter, 2022)
(Table [2](#tab:backg_st){reference-type="ref"
reference="tab:backg_st"}).

### In Situ Hybridization Imaging-Based Approaches

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
hundreds to thousands of genes. MERFISH (K. H. Chen et al., 2015) uses
combinatorial labelling and error-correcting barcodes to detect
thousands of RNA species in single cells.
[seqFISH]{acronym-label="seqFISH" acronym-form="singular+short"}+ (Eng
et al., 2019) leverages sequential rounds of hybridization with an
expanded pseudo-color palette, enabling detection of over 10,000 genes
at subcellular resolution.

Commercial platforms such as CosMx (Nanostring/Bruker Spatial) (K. H.
Chen et al., 2015), MERscope (Vizgen), and Xenium (10x Genomics)
(Janesick et al., 2023) implement these strategies to achieve
high-resolution imaging of hundreds to thousands of RNA species, often
with optional co-detection of proteins.

### In situ sequencing Imaging-Based Approaches

ISS techniques sequence RNA molecules directly within tissues,
preserving both spatial context and nucleotide identity. Unlike ISH, ISS
provides sequence information, enabling mutation and splice isoform
detection.

STARmap (Wang et al., 2018) improves detection efficiency by using DNA
nanostructures and hydrogel-tissue chemistry, while FISSEQ (Lee et al.,
2015), enables untargeted, whole-transcriptome analysis through in situ
reverse transcription and random-hexamer priming. These methods retain
spatial localization while offering a more detailed molecular readout
than hybridization alone.

Both ISH and ISS require fluorescence microscopy for imaging readouts
and are collectively referred to as imaging-based spatial
transcriptomics.

### Spatial Barcoding and Sequencing-Based Methods

Unlike imaging methods, spatial barcoding techniques rely on
sequencing-based detection. They use spatially encoded oligonucleotides
(barcodes) fixed to a surface (e.g., slide, bead, or grid) to capture
RNA from overlying tissue. After RNA capture, reverse transcription and
sequencing are performed, and spatial information is reconstructed based
on barcode identity.

Prominent examples include Visium (10X Genomics)(Ståhl et al., 2016),
which captures RNA on slide-mounted barcoded spots
($\sim$`<!-- -->`{=html}55 µm resolution), and it's new version, Visium
HD, that offers spatial resolution to $\sim$`<!-- -->`{=html}2--5 µm.
Other examples include Slide-seq (Rodriques et al., 2019) and
Slide-seqV2 (Stickels et al., 2021) that utilise barcoded beads with
known spatial locations ($\sim$`<!-- -->`{=html}10 µm resolution).

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

## Spatial metabolomics

Spatial metabolomics explores the spatial distribution of metabolites
directly in tissue sections, providing insight into biochemical activity
within the anatomical context (Aichler & Walch, 2015). The field is
largely driven by [MALDI-MSI]{acronym-label="MALDI-MSI"
acronym-form="singular+short"}(Stoeckli et al., 2001), a label-free,
untargeted technique capable of detecting a wide range of small
molecules including lipids, neurotransmitters, and drugs.

In MALDI-MSI, tissue sections are coated with a matrix that facilitates
ionization when hit by a laser. The resulting ions are analysed by mass
spectrometry to reconstruct spatial metabolite maps. MALDI-MSI offers
spatial resolution in the range of 10--$50\,\mu\text{m}$, with coverage
of hundreds to thousands of metabolites, depending on the tissue and
matrix.

## Spatial multi-omics

Spatial multi-omics technologies
(Table [3](#tab:backg_smultiomics){reference-type="ref"
reference="tab:backg_smultiomics"}) enable the simultaneous or
integrative profiling of multiple molecular layers---such as RNA,
proteins, and metabolites---within their spatial tissue context,
offering a more comprehensive understanding of cellular states and
interactions.

One of the most established platforms, GeoMx [DSP]{acronym-label="DSP"
acronym-form="singular+short"}(NanoString) (Merritt et al., 2020),
allows for high-plex profiling of both RNA and proteins within defined
regions of interest using oligonucleotide-tagged probes and UV-directed
barcode collection, though at limited spatial resolution
($\sim10–100\,\mu\text{m}$). Other advanced methods, such as
deterministic barcoding in tissue for spatial omics sequencing
(DBiT-seq) (Y. Liu et al., 2020), achieve co-detection of RNA and
proteins through microfluidic-based spatial barcoding on the same
section, offering high spatial resolution ($\sim 10\,\mu\text{m}$) and
true multimodal readouts. Similarly, Spatial-CITE-seq (Y. Liu et al.,
2023) adapts [CITE]{acronym-label="CITE" acronym-form="singular+short"}
to the spatial dimension, enabling the capture of transcriptomes
alongside surface protein markers.

Furthermore, MALDI-MSI have been successfully combined on the same slide
with both IMC (J. B. Nunes et al., 2024) and spatial transcriptomics
platforms like 10x Visium (Vicari et al., 2024).

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
and often obscured in bulk or single-modality data (Abadi et al., 2016;
Engblom & Lundeberg, 2025; Hsieh et al., 2022; Moses & Pachter, 2022).
They have also been used to study clonality differences in space
(Engblom et al., 2023; Giesen et al., 2014). As the applications are
vast and rapidly evolving, they are reviewed in detail elsewhere
(Engblom & Lundeberg, 2025; Hsieh et al., 2022; Moses & Pachter, 2022).

With this foundation in the underlying technologies and their biological
potential, the next section shifts focus to the analytical challenges
and computational frameworks required to interpret and integrate spatial
omics data.

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::: {#refs .references .csl-bib-body .hanging-indent entry-spacing="0" line-spacing="2"}
::: {#ref-P24_abadi_tensorflow_2016 .csl-entry}
Abadi, M., Agarwal, A., Barham, P., Brevdo, E., Chen, Z., Citro, C.,
Corrado, G. S., Davis, A., Dean, J., Devin, M., Ghemawat, S.,
Goodfellow, I., Harp, A., Irving, G., Isard, M., Jia, Y., Jozefowicz,
R., Kaiser, L., Kudlur, M., ... Zheng, X. (2016). *TensorFlow:
Large-Scale Machine Learning on Heterogeneous Distributed Systems*.
arXiv. <https://doi.org/10.48550/ARXIV.1603.04467>
:::

::: {#ref-B25_aebersold2016 .csl-entry}
Aebersold, R., & Mann, M. (2016). Mass-spectrometric exploration of
proteome structure and function. *Nature*, *537*(7620), 347--355.
<https://doi.org/10.1038/nature19949>
:::

::: {#ref-B62_maldi2015 .csl-entry}
Aichler, M., & Walch, A. (2015). MALDI Imaging mass spectrometry:
Current frontiers and perspectives in pathology research and practice.
*Laboratory Investigation*, *95*(4), 422--431.
<https://doi.org/10.1038/labinvest.2014.156>
:::

::: {#ref-B27_aldridge2020 .csl-entry}
Aldridge, S., & Teichmann, S. A. (2020). Single cell transcriptomics
comes of age. *Nature Communications*, *11*(1), 4307.
<https://doi.org/10.1038/s41467-020-18158-5>
:::

::: {#ref-B49asp2020 .csl-entry}
Asp, M., Bergenstråhle, J., & Lundeberg, J. (2020). Spatially Resolved
Transcriptomes---Next Generation Tools for Tissue Exploration.
*BioEssays*, *42*(10). <https://doi.org/10.1002/bies.201900221>
:::

::: {#ref-B22_baylin2011 .csl-entry}
Baylin, S. B., & Jones, P. A. (2011). A decade of exploring the cancer
epigenome --- biological and translational implications. *Nature Reviews
Cancer*, *11*(10), 726--734. <https://doi.org/10.1038/nrc3130>
:::

::: {#ref-B46_codex .csl-entry}
Black, S., Phillips, D., Hickey, J. W., Kennedy-Darling, J.,
Venkataraaman, V. G., Samusik, N., Goltsev, Y., Schürch, C. M., & Nolan,
G. P. (2021). CODEX multiplexed tissue imaging with DNA-conjugated
antibodies. *Nature Protocols*, *16*(8), 3802--3835.
:::

::: {#ref-B52_chen2015 .csl-entry}
Chen, K. H., Boettiger, A. N., Moffitt, J. R., Wang, S., & Zhuang, X.
(2015). Spatially resolved, highly multiplexed RNA profiling in single
cells. *Science*, *348*(6233), aaa6090.
<https://doi.org/10.1126/science.aaa6090>
:::

::: {#ref-B21_chen_mapping_2025 .csl-entry}
Chen, X., Xu, H., Shu, X., & Song, C.-X. (2025). Mapping epigenetic
modifications by sequencing technologies. *Cell Death &
Differentiation*, *32*(1), 56--65.
<https://doi.org/10.1038/s41418-023-01213-1>
:::

::: {#ref-B34_TCIA2013 .csl-entry}
Clark, K., Vendt, B., Smith, K., Freymann, J., Kirby, J., Koppel, P.,
Moore, S., Phillips, S., Maffitt, D., Pringle, M., Tarbox, L., & Prior,
F. (2013). The Cancer Imaging Archive (TCIA): Maintaining and Operating
a Public Information Repository. *Journal of Digital Imaging*, *26*(6),
1045--1057. <https://doi.org/10.1007/s10278-013-9622-7>
:::

::: {#ref-B50_du2023 .csl-entry}
Du, J., Yang, Y.-C., An, Z.-J., Zhang, M.-H., Fu, X.-H., Huang, Z.-F.,
Yuan, Y., & Hou, J. (2023). Advances in spatial transcriptomics and
related data analysis strategies. *Journal of Translational Medicine*,
*21*(1), 330. <https://doi.org/10.1186/s12967-023-04150-2>
:::

::: {#ref-B53_seqFISH+2019 .csl-entry}
Eng, C.-H. L., Lawson, M., Zhu, Q., Dries, R., Koulena, N., Takei, Y.,
Yun, J., Cronin, C., Karp, C., Yuan, G.-C., & Cai, L. (2019).
Transcriptome-scale super-resolved imaging in tissues by RNA
[seqFISH]{.nocase}+. *Nature*, *568*(7751), 235--239.
<https://doi.org/10.1038/s41586-019-1049-y>
:::

::: {#ref-B69_engblom2025 .csl-entry}
Engblom, C., & Lundeberg, J. (2025). Putting cancer immunotherapy into
spatial context in the clinic. *Nature Biotechnology*, *43*(4),
471--476. <https://doi.org/10.1038/s41587-025-02596-x>
:::

::: {#ref-B73_engblom_spatial_2023 .csl-entry}
Engblom, C., Thrane, K., Lin, Q., Andersson, A., Toosi, H., Chen, X.,
Steiner, E., Lu, C., Mantovani, G., Hagemann-Jensen, M., Saarenpää, S.,
Jangard, M., Saez-Rodriguez, J., Michaëlsson, J., Hartman, J.,
Lagergren, J., Mold, J. E., Lundeberg, J., & Frisén, J. (2023). Spatial
transcriptomics of B cell and T cell receptors reveals lymphocyte clonal
dynamics. *Science*, *382*(6675), eadf8486.
<https://doi.org/10.1126/science.adf8486>
:::

::: {#ref-B47_B72_S4_IMC2014 .csl-entry}
Giesen, C., Wang, H. A. O., Schapiro, D., Zivanovic, N., Jacobs, A.,
Hattendorf, B., Schüffler, P. J., Grolimund, D., Buhmann, J. M., Brandt,
S., Varga, Z., Wild, P. J., Günther, D., & Bodenmiller, B. (2014).
Highly multiplexed imaging of tumor tissues with subcellular resolution
by mass cytometry. *Nature Methods*, *11*(4), 417--422.
<https://doi.org/10.1038/nmeth.2869>
:::

::: {#ref-P5_B44_S2_CODEX2018 .csl-entry}
Goltsev, Y., Samusik, N., Kennedy-Darling, J., Bhate, S., Hale, M.,
Vazquez, G., Black, S., & Nolan, G. P. (2018). Deep Profiling of Mouse
Splenic Architecture with CODEX Multiplexed Imaging. *Cell*, *174*(4),
968--981.e15. <https://doi.org/10.1016/j.cell.2018.07.010>
:::

::: {#ref-B39_hickey2022 .csl-entry}
Hickey, J. W., Neumann, E. K., Radtke, A. J., Camarillo, J. M.,
Beuschel, R. T., Albanese, A., McDonough, E., Hatler, J., Wiblin, A. E.,
Fisher, J., Croteau, J., Small, E. C., Sood, A., Caprioli, R. M.,
Angelo, R. M., Nolan, G. P., Chung, K., Hewitt, S. M., Germain, R. N.,
... Saka, S. K. (2022). Spatial mapping of protein composition and
tissue organization: A primer for multiplexed antibody-based imaging.
*Nature Methods*, *19*(3), 284--295.
<https://doi.org/10.1038/s41592-021-01316-y>
:::

::: {#ref-B17_2018 .csl-entry}
Hoadley, K. A., Yau, C., Hinoue, T., Wolf, D. M., Lazar, A. J., Drill,
E., Shen, R., Taylor, A. M., Cherniack, A. D., Thorsson, V., Akbani, R.,
Bowlby, R., Wong, C. K., Wiznerowicz, M., Sanchez-Vega, F., Robertson,
A. G., Schneider, B. G., Lawrence, M. S., Noushmehr, H., ... Laird, P.
W. (2018). Cell-of-Origin Patterns Dominate the Molecular Classification
of 10,000 Tumors from 33 Types of Cancer. *Cell*, *173*(2), 291--304.e6.
<https://doi.org/10.1016/j.cell.2018.03.022>
:::

::: {#ref-B71_hsieh2022 .csl-entry}
Hsieh, W.-C., Budiarto, B. R., Wang, Y.-F., Lin, C.-Y., Gwo, M.-C., So,
D. K., Tzeng, Y.-S., & Chen, S.-Y. (2022). Spatial multi-omics analyses
of the tumor immune microenvironment. *Journal of Biomedical Science*,
*29*(1), 96. <https://doi.org/10.1186/s12929-022-00879-y>
:::

::: {#ref-B33_spatial_metab2023 .csl-entry}
Hu, T., Allam, M., Cai, S., Henderson, W., Yueh, B., Garipcan, A.,
Ievlev, A. V., Afkarian, M., Beyaz, S., & Coskun, A. F. (2023).
Single-cell spatial metabolomics with cell-type specific protein
profiling for tissue systems biology. *Nature Communications*, *14*(1),
8260. <https://doi.org/10.1038/s41467-023-43917-5>
:::

::: {#ref-B41_ijsselsteijn2019 .csl-entry}
Ijsselsteijn, M. E., Brouwer, T. P., Abdulrahman, Z., Reidy, E.,
Ramalheiro, A., Heeren, A. M., Vahrmeijer, A., Jordanova, E. S., & De
Miranda, N. F. (2019). Cancer immunophenotyping by seven‐colour
multispectral imaging without tyramide signal amplification. *The
Journal of Pathology: Clinical Research*, *5*(1), 3--11.
<https://doi.org/10.1002/cjp2.113>
:::

::: {#ref-B56_G5_xenium .csl-entry}
Janesick, A., Shelansky, R., Gottscho, A. D., Wagner, F., Williams, S.
R., Rouault, M., Beliakoff, G., Morrison, C. A., Oliveira, M. F.,
Sicherman, J. T., Kohlway, A., Abousoud, J., Drennon, T. Y., Mohabbat,
S. H., 10x Development Teams, & Taylor, S. E. B. (2023). High resolution
mapping of the tumor microenvironment using integrated single-cell,
spatial and in situ analysis. *Nature Communications*, *14*(1), 8353.
<https://doi.org/10.1038/s41467-023-43458-x>
:::

::: {#ref-B38_G1_jing2025 .csl-entry}
Jing, S., Wang, H., Lin, P., Yuan, J., Tang, Z., & Li, H. (2025).
Quantifying and interpreting biologically meaningful spatial signatures
within tumor microenvironments. *Npj Precision Oncology*, *9*(1), 68.
<https://doi.org/10.1038/s41698-025-00857-1>
:::

::: {#ref-B48_P4_keren_mibi-tof_2019 .csl-entry}
Keren, L., Bosse, M., Thompson, S., Risom, T., Vijayaragavan, K.,
McCaffrey, E., Marquez, D., Angoshtari, R., Greenwald, N. F., Fienberg,
H., Wang, J., Kambham, N., Kirkwood, D., Nolan, G., Montine, T. J.,
Galli, S. J., West, R., Bendall, S. C., & Angelo, M. (2019). MIBI-TOF: A
multiplexed imaging platform relates cellular phenotypes and tissue
structure. *Science Advances*, *5*(10), eaax5851.
<https://doi.org/10.1126/sciadv.aax5851>
:::

::: {#ref-B28_kinker2020 .csl-entry}
Kinker, G. S., Greenwald, A. C., Tal, R., Orlova, Z., Cuoco, M. S.,
McFarland, J. M., Warren, A., Rodman, C., Roth, J. A., Bender, S. A.,
Kumar, B., Rocco, J. W., Fernandes, P. A. C. M., Mader, C. C.,
Keren-Shaul, H., Plotnikov, A., Barr, H., Tsherniak, A.,
Rozenblatt-Rosen, O., ... Tirosh, I. (2020). Pan-cancer single-cell
RNA-seq identifies recurring programs of cellular heterogeneity. *Nature
Genetics*, *52*(11), 1208--1218.
<https://doi.org/10.1038/s41588-020-00726-6>
:::

::: {#ref-B58_FISSEQ2015 .csl-entry}
Lee, J. H., Daugharthy, E. R., Scheiman, J., Kalhor, R., Ferrante, T.
C., Terry, R., Turczyk, B. M., Yang, J. L., Lee, H. S., Aach, J., Zhang,
K., & Church, G. M. (2015). Fluorescent in situ sequencing (FISSEQ) of
RNA for gene expression profiling in intact cells and tissues. *Nature
Protocols*, *10*(3), 442--458. <https://doi.org/10.1038/nprot.2014.191>
:::

::: {#ref-B20_li .csl-entry}
Li, Y. (2021). Modern epigenetics methods in biological research.
*Methods (San Diego, Calif.)*, *187*, 104--113.
<https://doi.org/10.1016/j.ymeth.2020.06.022>
:::

::: {#ref-B42_Cycif2018 .csl-entry}
Lin, J.-R., Izar, B., Wang, S., Yapp, C., Mei, S., Shah, P. M.,
Santagata, S., & Sorger, P. K. (2018). Highly multiplexed
immunofluorescence imaging of human tissues and tumors using t-CyCIF and
conventional optical microscopes. *eLife*, *7*, e31657.
<https://doi.org/10.7554/eLife.31657>
:::

::: {#ref-B35_liu_spatiotemporal2024 .csl-entry}
Liu, L., Chen, A., Li, Y., Mulder, J., Heyn, H., & Xu, X. (2024).
Spatiotemporal omics for biology and medicine. *Cell*, *187*(17),
4488--4519. <https://doi.org/10.1016/j.cell.2024.07.040>
:::

::: {#ref-B66_CITESEQ_liu2023 .csl-entry}
Liu, Y., DiStasio, M., Su, G., Asashima, H., Enninful, A., Qin, X.,
Deng, Y., Nam, J., Gao, F., Bordignon, P., Cassano, M., Tomayko, M., Xu,
M., Halene, S., Craft, J. E., Hafler, D., & Fan, R. (2023). High-plex
protein and whole transcriptome co-mapping at cellular resolution with
spatial CITE-seq. *Nature Biotechnology*, *41*(10), 1405--1409.
<https://doi.org/10.1038/s41587-023-01676-0>
:::

::: {#ref-B65_liu2020 .csl-entry}
Liu, Y., Yang, M., Deng, Y., Su, G., Enninful, A., Guo, C. C., Tebaldi,
T., Zhang, D., Kim, D., Bai, Z., Norris, E., Pan, A., Li, J., Xiao, Y.,
Halene, S., & Fan, R. (2020). High-Spatial-Resolution Multi-Omics
Sequencing via Deterministic Barcoding in Tissue. *Cell*, *183*(6),
1665--1681.e18. <https://doi.org/10.1016/j.cell.2020.10.026>
:::

::: {#ref-B37_elundberg2019 .csl-entry}
Lundberg, E., & Borner, G. H. H. (2019). Spatial proteomics: A powerful
discovery tool for cell biology. *Nature Reviews Molecular Cell
Biology*, *20*(5), 285--302. <https://doi.org/10.1038/s41580-018-0094-y>
:::

::: {#ref-B40_IHC2019 .csl-entry}
Magaki, S., Hojat, S. A., Wei, B., So, A., & Yong, W. H. (2019). An
Introduction to the Performance of Immunohistochemistry. In W. H. Yong
(Ed.), *Biobanking* (Vol. 1897, pp. 289--298). Springer New York.
<https://doi.org/10.1007/978-1-4939-8935-5_25>
:::

::: {#ref-B29_mansuri2023 .csl-entry}
Mansuri, M. S., Williams, K., & Nairn, A. C. (2023). Uncovering biology
by single-cell proteomics. *Communications Biology*, *6*(1), 381.
<https://doi.org/10.1038/s42003-023-04635-2>
:::

::: {#ref-B64_merritt_multiplex_2020 .csl-entry}
Merritt, C. R., Ong, G. T., Church, S. E., Barker, K., Danaher, P.,
Geiss, G., Hoang, M., Jung, J., Liang, Y., McKay-Fleisch, J., Nguyen,
K., Norgaard, Z., Sorg, K., Sprague, I., Warren, C., Warren, S.,
Webster, P. J., Zhou, Z., Zollinger, D. R., ... Beechem, J. M. (2020).
Multiplex digital spatial profiling of proteins and RNA in fixed tissue.
*Nature Biotechnology*, *38*(5), 586--599.
<https://doi.org/10.1038/s41587-020-0472-9>
:::

::: {#ref-B51_museumST_2022 .csl-entry}
Moses, L., & Pachter, L. (2022). Museum of spatial transcriptomics.
*Nature Methods*, *19*(5), 534--546.
<https://doi.org/10.1038/s41592-022-01409-2>
:::

::: {#ref-B67_nunes2024 .csl-entry}
Nunes, J. B., Ijsselsteijn, M. E., Abdelaal, T., Ursem, R., Van Der
Ploeg, M., Giera, M., Everts, B., Mahfouz, A., Heijs, B., & De Miranda,
N. F. C. C. (2024). Integration of mass cytometry and mass spectrometry
imaging for spatially resolved single-cell metabolic profiling. *Nature
Methods*, *21*(10), 1796--1800.
<https://doi.org/10.1038/s41592-024-02392-6>
:::

::: {#ref-B19_nunes2024 .csl-entry}
Nunes, L., Li, F., Wu, M., Luo, T., Hammarström, K., Torell, E.,
Ljuslinder, I., Mezheyeuski, A., Edqvist, P.-H., Löfgren-Burström, A.,
Zingmark, C., Edin, S., Larsson, C., Mathot, L., Osterman, E.,
Osterlund, E., Ljungström, V., Neves, I., Yacoub, N., ... Sjöblom, T.
(2024). Prognostic genome and transcriptome signatures in colorectal
cancers. *Nature*, *633*(8028), 137--146.
<https://doi.org/10.1038/s41586-024-07769-3>
:::

::: {#ref-B36_palla_spatial_2022 .csl-entry}
Palla, G., Fischer, D. S., Regev, A., & Theis, F. J. (2022). Spatial
components of molecular tissue biology. *Nature Biotechnology*, *40*(3),
308--318. <https://doi.org/10.1038/s41587-021-01182-1>
:::

::: {#ref-B43_ibex_2022 .csl-entry}
Radtke, A. J., Chu, C. J., Yaniv, Z., Yao, L., Marr, J., Beuschel, R.
T., Ichise, H., Gola, A., Kabat, J., Lowekamp, B., Speranza, E.,
Croteau, J., Thakur, N., Jonigk, D., Davis, J. L., Hernandez, J. M., &
Germain, R. N. (2022). IBEX: An iterative immunolabeling and chemical
bleaching method for high-content imaging of diverse tissues. *Nature
Protocols*, *17*(2), 378--401.
<https://doi.org/10.1038/s41596-021-00644-9>
:::

::: {#ref-B60_slideseq2019 .csl-entry}
Rodriques, S. G., Stickels, R. R., Goeva, A., Martin, C. A., Murray, E.,
Vanderburg, C. R., Welch, J., Chen, L. M., Chen, F., & Macosko, E. Z.
(2019). Slide-seq: A scalable technology for measuring genome-wide
expression at high spatial resolution. *Science*, *363*(6434),
1463--1467. <https://doi.org/10.1126/science.aaw1219>
:::

::: {#ref-B45_isaber2019 .csl-entry}
Saka, S. K., Wang, Y., Kishi, J. Y., Zhu, A., Zeng, Y., Xie, W., Kirli,
K., Yapp, C., Cicconet, M., Beliveau, B. J., Lapan, S. W., Yin, S., Lin,
M., Boyden, E. S., Kaeser, P. S., Pihan, G., Church, G. M., & Yin, P.
(2019). Immuno-SABER enables highly multiplexed and amplified protein
imaging in tissues. *Nature Biotechnology*, *37*(9), 1080--1090.
<https://doi.org/10.1038/s41587-019-0207-y>
:::

::: {#ref-B26_schmidt2021 .csl-entry}
Schmidt, D. R., Patel, R., Kirsch, D. G., Lewis, C. A., Vander Heiden,
M. G., & Locasale, J. W. (2021). Metabolomics in cancer research and
emerging applications in clinical oncology. *CA:A Cancer Journal for
Clinicians*, *71*(4), 333--358. <https://doi.org/10.3322/caac.21670>
:::

::: {#ref-B31_setayesh2023 .csl-entry}
Setayesh, S. M., Ndacayisaba, L. J., Rappard, K. E., Hennes, V., Rueda,
L. Y. M., Tang, G., Lin, P., Orlowski, R. Z., Symer, D. E., Manasanch,
E. E., Shishido, S. N., & Kuhn, P. (2023). Targeted single-cell
proteomic analysis identifies new liquid biopsy biomarkers associated
with multiple myeloma. *Npj Precision Oncology*, *7*(1), 95.
<https://doi.org/10.1038/s41698-023-00446-0>
:::

::: {#ref-B32_seydel_metab_2021 .csl-entry}
Seydel, C. (2021). Single-cell metabolomics hits its stride. *Nature
Methods*, *18*(12), 1452--1456.
<https://doi.org/10.1038/s41592-021-01333-x>
:::

::: {#ref-B30_protchalg2023 .csl-entry}
Single-cell proteomics: Challenges and prospects. (2023). *Nature
Methods*, *20*(3), 317--318.
<https://doi.org/10.1038/s41592-023-01828-9>
:::

::: {#ref-B59_stahl2016 .csl-entry}
Ståhl, P. L., Salmén, F., Vickovic, S., Lundmark, A., Navarro, J. F.,
Magnusson, J., Giacomello, S., Asp, M., Westholm, J. O., Huss, M.,
Mollbrink, A., Linnarsson, S., Codeluppi, S., Borg, Å., Pontén, F.,
Costea, P. I., Sahlén, P., Mulder, J., Bergmann, O., ... Frisén, J.
(2016). Visualization and analysis of gene expression in tissue sections
by spatial transcriptomics. *Science*, *353*(6294), 78--82.
<https://doi.org/10.1126/science.aaf2403>
:::

::: {#ref-B61_slideseqv2_2021 .csl-entry}
Stickels, R. R., Murray, E., Kumar, P., Li, J., Marshall, J. L., Di
Bella, D. J., Arlotta, P., Macosko, E. Z., & Chen, F. (2021). Highly
sensitive spatial transcriptomics at near-cellular resolution with
Slide-[seqV2]{.nocase}. *Nature Biotechnology*, *39*(3), 313--319.
<https://doi.org/10.1038/s41587-020-0739-1>
:::

::: {#ref-B63_IMS2001 .csl-entry}
Stoeckli, M., Chaurand, P., Hallahan, D. E., & Caprioli, R. M. (2001).
Imaging mass spectrometry: A new technology for the analysis of protein
expression in mammalian tissues. *Nature Medicine*, *7*(4), 493--496.
<https://doi.org/10.1038/86573>
:::

::: {#ref-B24_suhre2021 .csl-entry}
Suhre, K., McCarthy, M. I., & Schwenk, J. M. (2021). Genetics meets
proteomics: Perspectives for large population-based studies. *Nature
Reviews Genetics*, *22*(1), 19--37.
<https://doi.org/10.1038/s41576-020-0268-2>
:::

::: {#ref-B18_TCGA2012 .csl-entry}
The Cancer Genome Atlas Network. (2012). Comprehensive molecular
characterization of human colon and rectal cancer. *Nature*,
*487*(7407), 330--337. <https://doi.org/10.1038/nature11252>
:::

::: {#ref-B16_TCGA2013 .csl-entry}
The Cancer Genome Atlas Research Network, Weinstein, J. N., Collisson,
E. A., Mills, G. B., Shaw, K. R. M., Ozenberger, B. A., Ellrott, K.,
Shmulevich, I., Sander, C., & Stuart, J. M. (2013). The Cancer Genome
Atlas Pan-Cancer analysis project. *Nature Genetics*, *45*(10),
1113--1120. <https://doi.org/10.1038/ng.2764>
:::

::: {#ref-B23_protmap2015 .csl-entry}
Uhlén, M., Fagerberg, L., Hallström, B. M., Lindskog, C., Oksvold, P.,
Mardinoglu, A., Sivertsson, Å., Kampf, C., Sjöstedt, E., Asplund, A.,
Olsson, I., Edlund, K., Lundberg, E., Navani, S., Szigyarto, C. A.-K.,
Odeberg, J., Djureinovic, D., Takanen, J. O., Hober, S., ... Pontén, F.
(2015). Tissue-based map of the human proteome. *Science*, *347*(6220),
1260419. <https://doi.org/10.1126/science.1260419>
:::

::: {#ref-B68_vicari2024 .csl-entry}
Vicari, M., Mirzazadeh, R., Nilsson, A., Shariatgorji, R., Bjärterot,
P., Larsson, L., Lee, H., Nilsson, M., Foyer, J., Ekvall, M.,
Czarnewski, P., Zhang, X., Svenningsson, P., Käll, L., Andrén, P. E., &
Lundeberg, J. (2024). Spatial multimodal analysis of transcriptomes and
metabolomes in tissues. *Nature Biotechnology*, *42*(7), 1046--1050.
<https://doi.org/10.1038/s41587-023-01937-y>
:::

::: {#ref-B57_STARmap2018 .csl-entry}
Wang, X., Allen, W. E., Wright, M. A., Sylwestrak, E. L., Samusik, N.,
Vesuna, S., Evans, K., Liu, C., Ramakrishnan, C., Liu, J., Nolan, G. P.,
Bava, F.-A., & Deisseroth, K. (2018). Three-dimensional intact-tissue
sequencing of single-cell transcriptional states. *Science (New York,
N.Y.)*, *361*(6400), eaat5691. <https://doi.org/10.1126/science.aat5691>
:::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
