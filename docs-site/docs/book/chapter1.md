Understanding the complexity of cancer, particularly its cellular
heterogeneity and dynamic microenvironment, relies heavily on advanced
analytical technologies.

Traditionally, cancer research has been dominated by genomic and
transcriptomic approaches, which have illuminated key mutations,
epigenetic alterations, and dysregulated pathways across tumour
types^1--4^. More recently, multi-omics strategies---including
epigenomics^5--7^, proteomics^8--10^, and metabolomics^11^ --- have
expanded our understanding of tumour biology by capturing additional
layers of molecular regulation.

Single-cell technologies have further revolutionized the field by
enabling the dissection of intra-tumoral heterogeneity at cellular
resolution, uncovering rare cell populations and lineage trajectories
that bulk analyses obscure^12,13^. These innovations have also extended
to single-cell proteomics^14--16^ and metabolomics^17,18^, offering
increasingly comprehensive insights into tumour ecosystems. However,
despite their power, these methods often lack spatial context---an
essential component for understanding how cellular localization and
cell--cell interactions within the tumour microenvironment influence
disease progression and treatment response.

On the other hand, spatial information is routinely obtained in the
clinic through histopathological staining (e.g., H&E),
[IHC]{acronym-label="IHC" acronym-form="singular+short"}, and
radiological imaging^19^. These approaches offer spatial resolution at
the tissue or organ level and remain vital for diagnosis and clinical
decision-making, but are inherently limited in molecular detail and
throughput, lacking the capacity to resolve cellular heterogeneity or
dynamic molecular interactions at scale.

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
detection of molecular constituents at exact spatial coordinates^20^.
However different techniques vary greatly in resolution, scale and
molecular complexity^21^.

## Spatial proteomics

Spatial proteomics encompasses technologies that enable the in-situ
profiling of proteins within tissues, preserving their spatial
localization. Most spatial proteomics techniques detect proteins using
antibodies tagged with fluorophores, metals, or DNA barcodes. These tags
are then read using technologies such as fluorescence microscopy, mass
spectrometry, or DNA-based imaging to map the spatial distribution of
proteins within tissues^20,22--24^. A summary of key features across
these technologies is provided in
Table [1](#tab:spatial_proteomics){reference-type="ref"
reference="tab:spatial_proteomics"}.

Immunohistochemistry (IHC) is one of the most established clinical tools
for protein detection^25^. It uses enzyme-linked antibodies to generate
a chromogenic signal visible under brightfield microscopy. While widely
available and routinely used in diagnostics, traditional IHC is
generally limited to detecting one or a few markers per tissue section,
making it unsuitable for high-dimensional spatial profiling.

Among the fluorescence-based approaches, [IF]{acronym-label="IF"
acronym-form="singular+short"}^26^ is widely used but limited in
multiplexing. More advanced cyclic immunofluorescence methods---such as
[t-CyCIF]{acronym-label="t-CyCIF" acronym-form="singular+short"}^27^ and
[IBEX]{acronym-label="IBEX" acronym-form="singular+short"}^28^ - use
iterative staining and imaging cycles to overcome spectral limitations,
enabling the detection of 40--60+ markers with spatial resolution down
to 200--300 nm.

DNA-barcoded approaches such as [CODEX]{acronym-label="CODEX"
acronym-form="singular+short"}^29^ and
[Immuno-SABER]{acronym-label="Immuno-SABER"
acronym-form="singular+short"}^30^ further enhance multiplexing. CODEX
utilizes DNA-barcoded antibodies and sequential hybridization of
fluorescent probes, achieving high multiplexing (up to 60 proteins) with
single-cell resolution ($\sim$`<!-- -->`{=html}500 nm) in a single
imaging plane^31^. Immuno-SABER employs orthogonal DNA concatemers for
signal amplification achieving 32-plex with same resolution as
COSMX^30^.

Mass spectrometry-based approaches---notably [IMC]{acronym-label="IMC"
acronym-form="singular+short"}^32^ and
[MIBI-TOF]{acronym-label="MIBI-TOF"
acronym-form="singular+short"}^33^---use antibodies conjugated to
isotopically pure lanthanide metals, which are detected using laser
ablation (IMC) or ion beams (MIBI-TOF). These methods avoid fluorescence
background and allow simultaneous quantification of 40--50 proteins per
tissue section. IMC offers spatial resolution of approximately 1 μm,
while MIBI achieves higher resolution ($\sim$`<!-- -->`{=html}300 nm),
albeit with more complex instrumentation.

::: {#tab:spatial_proteomics}

------------------------------------------------------------------------

**Type**\
**Capacity**\
**Scale**\
IHC\
(chromogen)\
Microscopy 1--2 $\sim$`<!-- -->`{=html}1--5 µm Cellular\
IF Fluorophore\
Microscopy 4--7 $\sim$`<!-- -->`{=html}200--500 nm Subcellular\
cyCIF / 4i / IBEX\
(iterative)\
Imaging $>$`<!-- -->`{=html}60 $\sim$`<!-- -->`{=html}250 nm
Subcellular\
CODEX DNA Barcode\
Microscopy $>$`<!-- -->`{=html}40--60+ $\sim$`<!-- -->`{=html}250 nm
Subcellular\
Immuno-SABER\
(concatemers)\
Microscopy $>$`<!-- -->`{=html}50+ $\sim$`<!-- -->`{=html}200--300 nm
Subcellular\
IMC Metal Isotopes\
+ MS $\sim$`<!-- -->`{=html}40 $\sim$`<!-- -->`{=html}1 µm Cellular\
MIBI Metal Isotopes\
+ MS $\sim$`<!-- -->`{=html}40 $\sim$`<!-- -->`{=html}300 nm
Subcellular\
------------------- -----------------------------
------------------------------------ ------------- -- --

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
throughput, and tissue compatibility^20,34--36^
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
hundreds to thousands of genes. MERFISH^37^ uses combinatorial labelling
and error-correcting barcodes to detect thousands of RNA species in
single cells. [seqFISH]{acronym-label="seqFISH"
acronym-form="singular+short"}+^38^ leverages sequential rounds of
hybridization with an expanded pseudo-color palette, enabling detection
of over 10,000 genes at subcellular resolution.

Commercial platforms such as CosMx (Nanostring/Bruker Spatial)^37^,
MERscope (Vizgen), and Xenium (10x Genomics)^39^ implement these
strategies to achieve high-resolution imaging of hundreds to thousands
of RNA species, often with optional co-detection of proteins.

### In situ sequencing Imaging-Based Approaches

ISS techniques sequence RNA molecules directly within tissues,
preserving both spatial context and nucleotide identity. Unlike ISH, ISS
provides sequence information, enabling mutation and splice isoform
detection.

STARmap^40^ improves detection efficiency by using DNA nanostructures
and hydrogel-tissue chemistry, while FISSEQ^41^, enables untargeted,
whole-transcriptome analysis through in situ reverse transcription and
random-hexamer priming. These methods retain spatial localization while
offering a more detailed molecular readout than hybridization alone.

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

Prominent examples include Visium (10X Genomics)^42^, which captures RNA
on slide-mounted barcoded spots ($\sim$`<!-- -->`{=html}55 µm
resolution), and it's new version, Visium HD, that offers spatial
resolution to $\sim$`<!-- -->`{=html}2--5 µm. Other examples include
Slide-seq^43^ and Slide-seqV2^44^ that utilise barcoded beads with known
spatial locations ($\sim$`<!-- -->`{=html}10 µm resolution).

These methods offer broader transcriptomic coverage---often approaching
the whole transcriptome---and are scalable to larger tissue sections.
However, they typically offer lower spatial resolution than
imaging-based platforms and may not reveal single-cell or subcellular
detail.

::: {#tab:backg_st}

------------------------------------------------------------------------

**Coverage**\
**Scale**\
smFISH ISH 1--10 genes $\sim$`<!-- -->`{=html}200--300 nm Subcellular
MERFISH ISH ,000+ genes $\sim$`<!-- -->`{=html}100--300 nm Subcellular

seqFISH+ ISH-based \>10,000 genes $\sim$`<!-- -->`{=html}200 nm
Subcellular CosMx ISH / smFISH $\sim$`<!-- -->`{=html}1,000 genes
$\sim$`<!-- -->`{=html}250 nm Subcellular MERscope MERFISH\
genes $\sim$`<!-- -->`{=html}300 nm Subcellular\
Xenium smFISH\
genes $\sim$`<!-- -->`{=html}280 nm Subcellular\
STARmap ISS\
genes $\sim$`<!-- -->`{=html}2--3$\mu\text{m}$ Single-cell\
FISSEQ ISS\
transcriptome $\sim$`<!-- -->`{=html}300 nm--1$\mu\text{m}$\
Subcellular\
Visium Spatial Barcoding (Slide)\
transcriptome $\sim$`<!-- -->`{=html}55$\mu\text{m}$ Multicellular\
Visium HD Spatial Barcoding (Slide)\
transcriptome $\sim$`<!-- -->`{=html}2--5$\mu\text{m}$ Close to
single-cell\
Slide-seq Spatial Barcoding (Beads)\
transcriptome $\sim$`<!-- -->`{=html}10$\mu\text{m}$ Cellular\
Slide-seqV2 Spatial Barcoding (Beads)\
transcriptome $\sim$`<!-- -->`{=html}10$\mu\text{m}$ Cellular\
--------------- -----------------------------------------------
------------------------------------
------------------------------------ -------------

: Comparison table of key features of spatial transcriptomics profiling
methods.
:::

## Spatial metabolomics

Spatial metabolomics explores the spatial distribution of metabolites
directly in tissue sections, providing insight into biochemical activity
within the anatomical context^45^. The field is largely driven by
[MALDI-MSI]{acronym-label="MALDI-MSI"
acronym-form="singular+short"}^46^, a label-free, untargeted technique
capable of detecting a wide range of small molecules including lipids,
neurotransmitters, and drugs.

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
acronym-form="singular+short"}(NanoString)^47^, allows for high-plex
profiling of both RNA and proteins within defined regions of interest
using oligonucleotide-tagged probes and UV-directed barcode collection,
though at limited spatial resolution ($\sim10–100\,\mu\text{m}$). Other
advanced methods, such as deterministic barcoding in tissue for spatial
omics sequencing (DBiT-seq)^48^, achieve co-detection of RNA and
proteins through microfluidic-based spatial barcoding on the same
section, offering high spatial resolution ($\sim 10\,\mu\text{m}$) and
true multimodal readouts. Similarly, Spatial-CITE-seq^49^ adapts
[CITE]{acronym-label="CITE" acronym-form="singular+short"} to the
spatial dimension, enabling the capture of transcriptomes alongside
surface protein markers.

Furthermore, MALDI-MSI have been successfully combined on the same slide
with both IMC^50^ and spatial transcriptomics platforms like 10x
Visium^51^.

::: {#tab:backg_smultiomics}

------------------------------------------------------------------------

**Targets**\
**Targets**\
**Coverage**\
**Resolution**\
GeoMx DSP $\sim$`<!-- -->`{=html}18,000 100+ --\
(ROI-based)\
DBiT-seq $\sim$`<!-- -->`{=html}6,000-10,000
$\sim$`<!-- -->`{=html}30-100 -- $\sim$`<!-- -->`{=html}10$\mu\text{m}$
Spatial-CITE-seq $\sim$`<!-- -->`{=html}5,000-10,000
$\sim$`<!-- -->`{=html}100-200 --
$\sim$`<!-- -->`{=html}20--50$\mu\text{m}$ MALDI-MSI + IMC --
$\sim$`<!-- -->`{=html}30-50\
1,000s $\sim$`<!-- -->`{=html}1--10$\mu\text{m}$\
MALDI-MSI + Visium 18000 -- 100-1000\
55$\mu\text{m}$ (ST)\
---------------------- -------------------------------------------
-------------------------------- ----------
--------------------------------------------

: Comparison table of key features of spatial multi-omics methods.
:::

Altogether, spatial omics are accelerating insights into tissue biology
and transforming multiple areas of medicine. In particular, they have
been extensively applied to study complex microenvironments, such as
tumours, where spatial context and cellular heterogeneity are critical
and often obscured in bulk or single-modality data^36,52--54^. They have
also been used to study clonality differences in space^32,55^. As the
applications are vast and rapidly evolving, they are reviewed in detail
elsewhere^36,53,54^.

With this foundation in the underlying technologies and their biological
potential, the next section shifts focus to the analytical challenges
and computational frameworks required to interpret and integrate spatial
omics data.

:::::::::::::::::::::::::::::::::::::::: {#refs .references .csl-bib-body entry-spacing="0" line-spacing="2"}
::: {#ref-B16_TCGA2013 .csl-entry}
[1.]{.csl-left-margin}[The Cancer Genome Atlas Research Network *et al.*
[The Cancer Genome Atlas Pan-Cancer analysis
project](https://doi.org/10.1038/ng.2764). *Nature Genetics* **45**,
1113--1120 (2013).]{.csl-right-inline}
:::

::: {#ref-B17_2018 .csl-entry}
[2.]{.csl-left-margin}[Hoadley, K. A. *et al.* [Cell-of-Origin Patterns
Dominate the Molecular Classification of 10,000 Tumors from 33 Types of
Cancer](https://doi.org/10.1016/j.cell.2018.03.022). *Cell* **173**,
291--304.e6 (2018).]{.csl-right-inline}
:::

::: {#ref-B18_TCGA2012 .csl-entry}
[3.]{.csl-left-margin}[The Cancer Genome Atlas Network. [Comprehensive
molecular characterization of human colon and rectal
cancer](https://doi.org/10.1038/nature11252). *Nature* **487**, 330--337
(2012).]{.csl-right-inline}
:::

::: {#ref-B19_nunes2024 .csl-entry}
[4.]{.csl-left-margin}[Nunes, L. *et al.* [Prognostic genome and
transcriptome signatures in colorectal
cancers](https://doi.org/10.1038/s41586-024-07769-3). *Nature* **633**,
137--146 (2024).]{.csl-right-inline}
:::

::: {#ref-B20_li .csl-entry}
[5.]{.csl-left-margin}[Li, Y. [Modern epigenetics methods in biological
research](https://doi.org/10.1016/j.ymeth.2020.06.022). *Methods (San
Diego, Calif.)* **187**, 104--113 (2021).]{.csl-right-inline}
:::

::: {#ref-B21_chen_mapping_2025 .csl-entry}
[6.]{.csl-left-margin}[Chen, X., Xu, H., Shu, X. & Song, C.-X. [Mapping
epigenetic modifications by sequencing
technologies](https://doi.org/10.1038/s41418-023-01213-1). *Cell Death &
Differentiation* **32**, 56--65 (2025).]{.csl-right-inline}
:::

::: {#ref-B22_baylin2011 .csl-entry}
[7.]{.csl-left-margin}[Baylin, S. B. & Jones, P. A. [A decade of
exploring the cancer epigenome --- biological and translational
implications](https://doi.org/10.1038/nrc3130). *Nature Reviews Cancer*
**11**, 726--734 (2011).]{.csl-right-inline}
:::

::: {#ref-B23_protmap2015 .csl-entry}
[8.]{.csl-left-margin}[Uhlén, M. *et al.* [Tissue-based map of the human
proteome](https://doi.org/10.1126/science.1260419). *Science* **347**,
1260419 (2015).]{.csl-right-inline}
:::

::: {#ref-B24_suhre2021 .csl-entry}
[9.]{.csl-left-margin}[Suhre, K., McCarthy, M. I. & Schwenk, J. M.
[Genetics meets proteomics: Perspectives for large population-based
studies](https://doi.org/10.1038/s41576-020-0268-2). *Nature Reviews
Genetics* **22**, 19--37 (2021).]{.csl-right-inline}
:::

::: {#ref-B25_aebersold2016 .csl-entry}
[10.]{.csl-left-margin}[Aebersold, R. & Mann, M. [Mass-spectrometric
exploration of proteome structure and
function](https://doi.org/10.1038/nature19949). *Nature* **537**,
347--355 (2016).]{.csl-right-inline}
:::

::: {#ref-B26_schmidt2021 .csl-entry}
[11.]{.csl-left-margin}[Schmidt, D. R. *et al.* [Metabolomics in cancer
research and emerging applications in clinical
oncology](https://doi.org/10.3322/caac.21670). *CA:A Cancer Journal for
Clinicians* **71**, 333--358 (2021).]{.csl-right-inline}
:::

::: {#ref-B27_aldridge2020 .csl-entry}
[12.]{.csl-left-margin}[Aldridge, S. & Teichmann, S. A. [Single cell
transcriptomics comes of
age](https://doi.org/10.1038/s41467-020-18158-5). *Nature
Communications* **11**, 4307 (2020).]{.csl-right-inline}
:::

::: {#ref-B28_kinker2020 .csl-entry}
[13.]{.csl-left-margin}[Kinker, G. S. *et al.* [Pan-cancer single-cell
RNA-seq identifies recurring programs of cellular
heterogeneity](https://doi.org/10.1038/s41588-020-00726-6). *Nature
Genetics* **52**, 1208--1218 (2020).]{.csl-right-inline}
:::

::: {#ref-B29_mansuri2023 .csl-entry}
[14.]{.csl-left-margin}[Mansuri, M. S., Williams, K. & Nairn, A. C.
[Uncovering biology by single-cell
proteomics](https://doi.org/10.1038/s42003-023-04635-2). *Communications
Biology* **6**, 381 (2023).]{.csl-right-inline}
:::

::: {#ref-B30_protchalg2023 .csl-entry}
[15.]{.csl-left-margin}[[Single-cell proteomics: Challenges and
prospects](https://doi.org/10.1038/s41592-023-01828-9). *Nature Methods*
**20**, 317--318 (2023).]{.csl-right-inline}
:::

::: {#ref-B31_setayesh2023 .csl-entry}
[16.]{.csl-left-margin}[Setayesh, S. M. *et al.* [Targeted single-cell
proteomic analysis identifies new liquid biopsy biomarkers associated
with multiple myeloma](https://doi.org/10.1038/s41698-023-00446-0). *npj
Precision Oncology* **7**, 95 (2023).]{.csl-right-inline}
:::

::: {#ref-B32_seydel_metab_2021 .csl-entry}
[17.]{.csl-left-margin}[Seydel, C. [Single-cell metabolomics hits its
stride](https://doi.org/10.1038/s41592-021-01333-x). *Nature Methods*
**18**, 1452--1456 (2021).]{.csl-right-inline}
:::

::: {#ref-B33_spatial_metab2023 .csl-entry}
[18.]{.csl-left-margin}[Hu, T. *et al.* [Single-cell spatial
metabolomics with cell-type specific protein profiling for tissue
systems biology](https://doi.org/10.1038/s41467-023-43917-5). *Nature
Communications* **14**, 8260 (2023).]{.csl-right-inline}
:::

::: {#ref-B34_TCIA2013 .csl-entry}
[19.]{.csl-left-margin}[Clark, K. *et al.* [The Cancer Imaging Archive
(TCIA): Maintaining and Operating a Public Information
Repository](https://doi.org/10.1007/s10278-013-9622-7). *Journal of
Digital Imaging* **26**, 1045--1057 (2013).]{.csl-right-inline}
:::

::: {#ref-B35_liu_spatiotemporal2024 .csl-entry}
[20.]{.csl-left-margin}[Liu, L. *et al.* [Spatiotemporal omics for
biology and medicine](https://doi.org/10.1016/j.cell.2024.07.040).
*Cell* **187**, 4488--4519 (2024).]{.csl-right-inline}
:::

::: {#ref-B36_palla_spatial_2022 .csl-entry}
[21.]{.csl-left-margin}[Palla, G., Fischer, D. S., Regev, A. & Theis, F.
J. [Spatial components of molecular tissue
biology](https://doi.org/10.1038/s41587-021-01182-1). *Nature
Biotechnology* **40**, 308--318 (2022).]{.csl-right-inline}
:::

::: {#ref-B37_elundberg2019 .csl-entry}
[22.]{.csl-left-margin}[Lundberg, E. & Borner, G. H. H. [Spatial
proteomics: A powerful discovery tool for cell
biology](https://doi.org/10.1038/s41580-018-0094-y). *Nature Reviews
Molecular Cell Biology* **20**, 285--302 (2019).]{.csl-right-inline}
:::

::: {#ref-B38_G1_jing2025 .csl-entry}
[23.]{.csl-left-margin}[Jing, S. *et al.* [Quantifying and interpreting
biologically meaningful spatial signatures within tumor
microenvironments](https://doi.org/10.1038/s41698-025-00857-1). *npj
Precision Oncology* **9**, 68 (2025).]{.csl-right-inline}
:::

::: {#ref-B39_hickey2022 .csl-entry}
[24.]{.csl-left-margin}[Hickey, J. W. *et al.* [Spatial mapping of
protein composition and tissue organization: A primer for multiplexed
antibody-based imaging](https://doi.org/10.1038/s41592-021-01316-y).
*Nature Methods* **19**, 284--295 (2022).]{.csl-right-inline}
:::

::: {#ref-B40_IHC2019 .csl-entry}
[25.]{.csl-left-margin}[Magaki, S., Hojat, S. A., Wei, B., So, A. &
Yong, W. H. [An Introduction to the Performance of
Immunohistochemistry](https://doi.org/10.1007/978-1-4939-8935-5_25). in
*Biobanking* (ed. Yong, W. H.) vol. 1897 289--298 (Springer New York,
New York, NY, 2019).]{.csl-right-inline}
:::

::: {#ref-B41_ijsselsteijn2019 .csl-entry}
[26.]{.csl-left-margin}[Ijsselsteijn, M. E. *et al.* [Cancer
immunophenotyping by seven‐colour multispectral imaging without tyramide
signal amplification](https://doi.org/10.1002/cjp2.113). *The Journal of
Pathology: Clinical Research* **5**, 3--11 (2019).]{.csl-right-inline}
:::

::: {#ref-B42_Cycif2018 .csl-entry}
[27.]{.csl-left-margin}[Lin, J.-R. *et al.* [Highly multiplexed
immunofluorescence imaging of human tissues and tumors using t-CyCIF and
conventional optical microscopes](https://doi.org/10.7554/eLife.31657).
*eLife* **7**, e31657 (2018).]{.csl-right-inline}
:::

::: {#ref-B43_ibex_2022 .csl-entry}
[28.]{.csl-left-margin}[Radtke, A. J. *et al.* [IBEX: An iterative
immunolabeling and chemical bleaching method for high-content imaging of
diverse tissues](https://doi.org/10.1038/s41596-021-00644-9). *Nature
Protocols* **17**, 378--401 (2022).]{.csl-right-inline}
:::

::: {#ref-P5_B44_S2_CODEX2018 .csl-entry}
[29.]{.csl-left-margin}[Goltsev, Y. *et al.* [Deep Profiling of Mouse
Splenic Architecture with CODEX Multiplexed
Imaging](https://doi.org/10.1016/j.cell.2018.07.010). *Cell* **174**,
968--981.e15 (2018).]{.csl-right-inline}
:::

::: {#ref-B45_isaber2019 .csl-entry}
[30.]{.csl-left-margin}[Saka, S. K. *et al.* [Immuno-SABER enables
highly multiplexed and amplified protein imaging in
tissues](https://doi.org/10.1038/s41587-019-0207-y). *Nature
Biotechnology* **37**, 1080--1090 (2019).]{.csl-right-inline}
:::

::: {#ref-B46_codex .csl-entry}
[31.]{.csl-left-margin}[Black, S. *et al.* CODEX multiplexed tissue
imaging with DNA-conjugated antibodies. *Nature protocols* **16**,
3802--3835 (2021).]{.csl-right-inline}
:::

::: {#ref-B47_B72_S4_IMC2014 .csl-entry}
[32.]{.csl-left-margin}[Giesen, C. *et al.* [Highly multiplexed imaging
of tumor tissues with subcellular resolution by mass
cytometry](https://doi.org/10.1038/nmeth.2869). *Nature Methods* **11**,
417--422 (2014).]{.csl-right-inline}
:::

::: {#ref-B48_P4_keren_mibi-tof_2019 .csl-entry}
[33.]{.csl-left-margin}[Keren, L. *et al.* [MIBI-TOF: A multiplexed
imaging platform relates cellular phenotypes and tissue
structure](https://doi.org/10.1126/sciadv.aax5851). *Science Advances*
**5**, eaax5851 (2019).]{.csl-right-inline}
:::

::: {#ref-B49asp2020 .csl-entry}
[34.]{.csl-left-margin}[Asp, M., Bergenstråhle, J. & Lundeberg, J.
[Spatially Resolved Transcriptomes---Next Generation Tools for Tissue
Exploration](https://doi.org/10.1002/bies.201900221). *BioEssays*
**42**, (2020).]{.csl-right-inline}
:::

::: {#ref-B50_du2023 .csl-entry}
[35.]{.csl-left-margin}[Du, J. *et al.* [Advances in spatial
transcriptomics and related data analysis
strategies](https://doi.org/10.1186/s12967-023-04150-2). *Journal of
Translational Medicine* **21**, 330 (2023).]{.csl-right-inline}
:::

::: {#ref-B51_museumST_2022 .csl-entry}
[36.]{.csl-left-margin}[Moses, L. & Pachter, L. [Museum of spatial
transcriptomics](https://doi.org/10.1038/s41592-022-01409-2). *Nature
Methods* **19**, 534--546 (2022).]{.csl-right-inline}
:::

::: {#ref-B52_chen2015 .csl-entry}
[37.]{.csl-left-margin}[Chen, K. H., Boettiger, A. N., Moffitt, J. R.,
Wang, S. & Zhuang, X. [Spatially resolved, highly multiplexed RNA
profiling in single cells](https://doi.org/10.1126/science.aaa6090).
*Science* **348**, aaa6090 (2015).]{.csl-right-inline}
:::

::: {#ref-B53_seqFISH+2019 .csl-entry}
[38.]{.csl-left-margin}[Eng, C.-H. L. *et al.* [Transcriptome-scale
super-resolved imaging in tissues by RNA
[seqFISH]{.nocase}+](https://doi.org/10.1038/s41586-019-1049-y).
*Nature* **568**, 235--239 (2019).]{.csl-right-inline}
::::::::::::::::::::::::::::::::::::::::

::: {#ref-B56_G5_xenium .csl-entry}
[39.]{.csl-left-margin}[Janesick, A. *et al.* [High resolution mapping
of the tumor microenvironment using integrated single-cell, spatial and
in situ analysis](https://doi.org/10.1038/s41467-023-43458-x). *Nature
Communications* **14**, 8353 (2023).]{.csl-right-inline}
:::

::: {#ref-B57_STARmap2018 .csl-entry}
[40.]{.csl-left-margin}[Wang, X. *et al.* [Three-dimensional
intact-tissue sequencing of single-cell transcriptional
states](https://doi.org/10.1126/science.aat5691). *Science (New York,
N.Y.)* **361**, eaat5691 (2018).]{.csl-right-inline}
:::

::: {#ref-B58_FISSEQ2015 .csl-entry}
[41.]{.csl-left-margin}[Lee, J. H. *et al.* [Fluorescent in situ
sequencing (FISSEQ) of RNA for gene expression profiling in intact cells
and tissues](https://doi.org/10.1038/nprot.2014.191). *Nature Protocols*
**10**, 442--458 (2015).]{.csl-right-inline}
:::

::: {#ref-B59_stahl2016 .csl-entry}
[42.]{.csl-left-margin}[Ståhl, P. L. *et al.* [Visualization and
analysis of gene expression in tissue sections by spatial
transcriptomics](https://doi.org/10.1126/science.aaf2403). *Science*
**353**, 78--82 (2016).]{.csl-right-inline}
:::

::: {#ref-B60_slideseq2019 .csl-entry}
[43.]{.csl-left-margin}[Rodriques, S. G. *et al.* [Slide-seq: A scalable
technology for measuring genome-wide expression at high spatial
resolution](https://doi.org/10.1126/science.aaw1219). *Science* **363**,
1463--1467 (2019).]{.csl-right-inline}
:::

::: {#ref-B61_slideseqv2_2021 .csl-entry}
[44.]{.csl-left-margin}[Stickels, R. R. *et al.* [Highly sensitive
spatial transcriptomics at near-cellular resolution with
Slide-[seqV2]{.nocase}](https://doi.org/10.1038/s41587-020-0739-1).
*Nature Biotechnology* **39**, 313--319 (2021).]{.csl-right-inline}
:::

::: {#ref-B62_maldi2015 .csl-entry}
[45.]{.csl-left-margin}[Aichler, M. & Walch, A. [MALDI Imaging mass
spectrometry: Current frontiers and perspectives in pathology research
and practice](https://doi.org/10.1038/labinvest.2014.156). *Laboratory
Investigation* **95**, 422--431 (2015).]{.csl-right-inline}
:::

::: {#ref-B63_IMS2001 .csl-entry}
[46.]{.csl-left-margin}[Stoeckli, M., Chaurand, P., Hallahan, D. E. &
Caprioli, R. M. [Imaging mass spectrometry: A new technology for the
analysis of protein expression in mammalian
tissues](https://doi.org/10.1038/86573). *Nature Medicine* **7**,
493--496 (2001).]{.csl-right-inline}
:::

::: {#ref-B64_merritt_multiplex_2020 .csl-entry}
[47.]{.csl-left-margin}[Merritt, C. R. *et al.* [Multiplex digital
spatial profiling of proteins and RNA in fixed
tissue](https://doi.org/10.1038/s41587-020-0472-9). *Nature
Biotechnology* **38**, 586--599 (2020).]{.csl-right-inline}
:::

::: {#ref-B65_liu2020 .csl-entry}
[48.]{.csl-left-margin}[Liu, Y. *et al.* [High-Spatial-Resolution
Multi-Omics Sequencing via Deterministic Barcoding in
Tissue](https://doi.org/10.1016/j.cell.2020.10.026). *Cell* **183**,
1665--1681.e18 (2020).]{.csl-right-inline}
:::

::: {#ref-B66_CITESEQ_liu2023 .csl-entry}
[49.]{.csl-left-margin}[Liu, Y. *et al.* [High-plex protein and whole
transcriptome co-mapping at cellular resolution with spatial
CITE-seq](https://doi.org/10.1038/s41587-023-01676-0). *Nature
Biotechnology* **41**, 1405--1409 (2023).]{.csl-right-inline}
:::

::: {#ref-B67_nunes2024 .csl-entry}
[50.]{.csl-left-margin}[Nunes, J. B. *et al.* [Integration of mass
cytometry and mass spectrometry imaging for spatially resolved
single-cell metabolic
profiling](https://doi.org/10.1038/s41592-024-02392-6). *Nature Methods*
**21**, 1796--1800 (2024).]{.csl-right-inline}
:::

::: {#ref-B68_vicari2024 .csl-entry}
[51.]{.csl-left-margin}[Vicari, M. *et al.* [Spatial multimodal analysis
of transcriptomes and metabolomes in
tissues](https://doi.org/10.1038/s41587-023-01937-y). *Nature
Biotechnology* **42**, 1046--1050 (2024).]{.csl-right-inline}
:::

::: {#ref-P24_abadi_tensorflow_2016 .csl-entry}
[52.]{.csl-left-margin}[Abadi, M. *et al.* TensorFlow: Large-Scale
Machine Learning on Heterogeneous Distributed Systems. (2016)
doi:[10.48550/ARXIV.1603.04467](https://doi.org/10.48550/ARXIV.1603.04467).]{.csl-right-inline}
:::

::: {#ref-B69_engblom2025 .csl-entry}
[53.]{.csl-left-margin}[Engblom, C. & Lundeberg, J. [Putting cancer
immunotherapy into spatial context in the
clinic](https://doi.org/10.1038/s41587-025-02596-x). *Nature
Biotechnology* **43**, 471--476 (2025).]{.csl-right-inline}
:::

::: {#ref-B71_hsieh2022 .csl-entry}
[54.]{.csl-left-margin}[Hsieh, W.-C. *et al.* [Spatial multi-omics
analyses of the tumor immune
microenvironment](https://doi.org/10.1186/s12929-022-00879-y). *Journal
of Biomedical Science* **29**, 96 (2022).]{.csl-right-inline}
:::

::: {#ref-B73_engblom_spatial_2023 .csl-entry}
[55.]{.csl-left-margin}[Engblom, C. *et al.* [Spatial transcriptomics of
B cell and T cell receptors reveals lymphocyte clonal
dynamics](https://doi.org/10.1126/science.adf8486). *Science* **382**,
eadf8486 (2023).]{.csl-right-inline}
:::

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
