# Spatially Resolved Omics Techniques

Spatial omics technologies have the potential to advance our
understanding of tumour ecosystems and improve clinical outcomes. The
essence of spatial omics lies in its aptitude for the simultaneous
detection of molecular constituents at exact spatial coordinates [1].
However different techniques vary greatly in resolution, scale and
molecular complexity [2].

## Spatial proteomics

Spatial proteomics encompasses technologies that enable the in-situ
profiling of proteins within tissues, preserving their spatial
localization. Most spatial proteomics techniques detect proteins using
antibodies tagged with fluorophores, metals, or DNA barcodes. These tags
are then read using technologies such as fluorescence microscopy, mass
spectrometry, or DNA-based imaging to map the spatial distribution of
proteins within tissues [1-5]. A summary of key
features across these technologies is provided in
Table1.

Immunohistochemistry (IHC) is one of the most established clinical tools
for protein detection [6]. It uses enzyme-linked antibodies to
generate a chromogenic signal visible under brightfield microscopy.
While widely available and routinely used in diagnostics, traditional
IHC is generally limited to detecting one or a few markers per tissue
section, making it unsuitable for high-dimensional spatial profiling.

Among the fluorescence-based approaches, Immunofluorescence (IF) [7] is widely
used but limited in multiplexing. More advanced cyclic immunofluorescence methods,
such as tissue-based cyclic immunofluorescence (t-CyCIF) [8] and IBEX [9], use iterative staining and imaging cycles to
overcome spectral limitations, enabling the detection of 40-60+ markers with spatial
resolution down to 200-300 nm.

DNA-barcoded approaches such as CODEX [10] and Immuno-SABER [11] further enhance
multiplexing. CODEX utilizes DNA-barcoded antibodies and sequential hybridization of
fluorescent probes, achieving high multiplexing (up to 60 proteins) with
single-cell resolution (∼500 nm) in a single imaging plane [12].
Immuno-SABER employs orthogonal DNA concatemers for signal amplification achieving
32-plex with same resolution as COSMX [11].

Mass spectrometry-based approaches, notably Imaging Mass Cytometry (IMC) [13]
and multiplexed ion beam imaging by time of flight (MIBI-TOF) [14], use antibodies
conjugated to isotopically pure lanthanide metals, which are detected using laser
ablation (IMC) or ion beams (MIBI-TOF). These methods avoid fluorescence
background and allow simultaneous quantification of 40-50 proteins per
tissue section. IMC offers spatial resolution of approximately 1 μm,
while MIBI achieves higher resolution (∼300 nm),
albeit with more complex instrumentation.


Table1-Comparison table of key features of spatial proteomics profiling methods.

| **Method** | **Tag** <br> **Type** | **Detection** | **Multiplexing** <br> **Capacity** | **Resolution** | **Resolution** <br> **Scale** |
| :--------- | :-------------------- | :------------ | :--------------------------------: | :------------: | :--------------------------: |
| IHC        | Enzyme<br>(chromogen) | Brightfield<br>Microscopy | 1--2                               | ~1--5 µm       | Cellular                     |
| IF         | Fluorophore           | Fluorescence<br>Microscopy | 4--7                               | ~200--500 nm   | Subcellular                  |
| cyCIF / 4i / IBEX | Fluorophore<br>(iterative) | Cyclic Fluorescence<br>Imaging | >60                                | ~250 nm        | Subcellular                  |
| CODEX      | DNA Barcode           | Fluorescence<br>Microscopy | >40--60+                           | ~250 nm        | Subcellular                  |
| Immuno-SABER | DNA Barcode<br>(concatemers) | Fluorescence<br>Microscopy | >50+                               | ~200--300 nm   | Subcellular                  |
| IMC        | Metal Isotopes        | Laser Ablation<br>+ MS | ~40                                | ~1 µm          | Cellular                     |
| MIBI       | Metal Isotopes        | Ion Beam<br>+ MS | ~40                                | ~300 nm        | Subcellular                  |

Together, these spatial proteomics platforms offer complementary
advantages in terms of marker throughput, resolution, and sensitivity,
enabling comprehensive characterization of the tumour microenvironment,
immune cell infiltration, and cellular architecture in situ.

## Spatial transcriptomics

Spatial transcriptomics enables the study of gene expression within the
tissue architecture, preserving spatial context at cellular or
subcellular resolution. These platforms can be broadly divided into two
main categories based on detection strategy: imaging-based
methods, including n Situ Hybridization (ISH) and In Situ Sequencing (ISS), 
and spatial barcoding methods, which rely on capture-based approaches followed
by sequencing. Each approach presents trade-offs in terms of resolution,
transcriptome coverage, throughput, and tissue compatibility [1, 15-17]

### In Situ Hybridization Imaging-Based Approaches

ISH-based techniques use fluorescently labelled probes that hybridize
directly to target RNA molecules in fixed tissue sections. These methods
are highly accurate and can offer single-molecule and single-cell
resolution, but typically have limitations in multiplexing capacity
unless cyclic imaging or barcode strategies are used.

Basic single molecule FISH (smFISH) can detect individual transcripts with
high spatial precision but is limited in the number of genes detectable due
to the finite number of distinguishable fluorophores.

Advanced multiplexed methods, such as Multiplexed error-robust FISH (MERFISH),
and seqFISH+, overcome these limits by iterative cycles of hybridization and
imaging, enabling detection of hundreds to thousands of genes. MERFISH [18]
uses combinatorial labelling and error-correcting barcodes to detect thousands of RNA
species in single cells. Sequential fluorescence in situ hybridization (seqFISH)+ [19]
leverages sequential rounds of hybridization with an expanded pseudo-color palette,
enabling detection of over 10,000 genes at subcellular resolution.

Commercial platforms such as CosMx (Nanostring/Bruker Spatial) [18],
MERscope (Vizgen), and Xenium (10x Genomics) [20] implement these
strategies to achieve high-resolution imaging of hundreds to thousands
of RNA species, often with optional co-detection of proteins.

### In situ sequencing Imaging-Based Approaches

ISS techniques sequence RNA molecules directly within tissues,
preserving both spatial context and nucleotide identity. Unlike ISH, ISS
provides sequence information, enabling mutation and splice isoform
detection.

STARmap [21] improves detection efficiency by using DNA nanostructures
and hydrogel-tissue chemistry, while FISSEQ [22], enables untargeted,
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

Prominent examples include Visium (10X Genomics) [23], which captures
RNA on slide-mounted barcoded spots (∼55 µm resolution), and it's new 
version, Visium HD, that offers spatial resolution to ∼2-5 µm. Other examples include
Slide-seq [24] and Slide-seqV2 [25] that utilise barcoded beads with
known spatial locations (∼10 µm resolution).

These methods offer broader transcriptomic coverage-often approaching
the whole transcriptome-and are scalable to larger tissue sections.
However, they typically offer lower spatial resolution than
imaging-based platforms and may not reveal single-cell or subcellular
detail.

Comparison table of key features of spatial transcriptomics profiling methods.

| **Method** | **Category** | **Transcript** <br> **Coverage** | **Resolution** | **Resolution** <br> **Scale** |
| :--------- | :----------- | :-------------------------------: | :------------: | :----------------------------- |
| smFISH     | ISH          | 1–10 genes                        | ~200–300 nm    | Subcellular                    |
| MERFISH    | ISH          | 1,000+ genes                      | ~100–300 nm    | Subcellular                    |
| seqFISH <br> seqFISH+ | ISH-based    | >10,000 genes                     | ~200 nm        | Subcellular                    |
| CosMx      | ISH / smFISH | ~1,000 genes                      | ~250 nm        | Subcellular                    |
| MERscope   | MERFISH      | ~500–1,000 <br> genes             | ~300 nm        | Subcellular                    |
| Xenium     | smFISH       | ~300–400 <br> genes               | ~280 nm        | Subcellular                    |
| STARmap    | ISS          | ~1,000 <br> genes                 | ~2–3 µm        | Single-cell                    |
| FISSEQ     | ISS          | Whole <br> transcriptome          | ~300 nm–1 µm   | Single-cell / <br> Subcellular |
| Visium     | Spatial Barcoding (Slide) | Whole <br> transcriptome          | ~55 µm         | Multicellular                  |
| Visium HD  | Spatial Barcoding (Slide) | Whole <br> transcriptome          | ~2–5 µm        | Close to single-cell           |
| Slide-seq  | Spatial Barcoding (Beads) | Whole <br> transcriptome          | ~10 µm         | Cellular                       |
| Slide-seqV2 | Spatial Barcoding (Beads) | Whole <br> transcriptome          | ~10 µm         | Cellular                       |

## Spatial metabolomics

Spatial metabolomics explores the spatial distribution of metabolites
directly in tissue sections, providing insight into biochemical activity
within the anatomical context [26]. The field is largely driven by
Matrix-Assisted Laser Desorption-Ionization Mass Spectrometry Imaging (MALDI-MSI) [27],
a label-free, untargeted technique capable of detecting a wide range of small molecules
including lipids, neurotransmitters, and drugs.

In MALDI-MSI, tissue sections are coated with a matrix that facilitates
ionization when hit by a laser. The resulting ions are analysed by mass
spectrometry to reconstruct spatial metabolite maps. MALDI-MSI offers
spatial resolution in the range of 10--50 um, with coverage
of hundreds to thousands of metabolites, depending on the tissue and
matrix.

## Spatial multi-omics

Spatial multi-omics technologies (Table 3) enable the simultaneous or
integrative profiling of multiple molecular layers-such as RNA,
proteins, and metabolites- within their spatial tissue context,
offering a more comprehensive understanding of cellular states and
interactions.

One of the most established platforms, GeoMx Digital Spatial Profiler 
(DSP)(NanoString)[28], allows for high-plex profiling of both RNA and
proteins within defined regions of interest using oligonucleotide-tagged
probes and UV-directed barcode collection, though at limited spatial 
resolution (∼10–100um). Other advanced methods, such as deterministic 
barcoding in tissue for spatial omics sequencing (DBiT-seq) [29],
achieve co-detection of RNA and proteins through microfluidic-based
spatial barcoding on the same section, offering high spatial resolution (∼10um)
and true multimodal readouts. Similarly, Spatial-CITE-seq [30] adapts
co-indexing of transcriptomes and epitopes (CITE) to the  spatial dimension,
enabling the capture of transcriptomes alongside surface protein markers.

Furthermore, MALDI-MSI have been successfully combined on the same slide
with both IMC [31] and spatial transcriptomics platforms like 10x Visium [32].


Comparison table of key features of spatial multi-omics methods.

| **Method** | **RNA** <br> **Targets** | **Protein** <br> **Targets** | **Metabolite** <br> **Coverage** | **Spatial** <br> **Resolution** |
| :--------- | :---------------------: | :--------------------------: | :------------------------------: | :-------------------------------------------: |
| GeoMx DSP  | ~18,000                 | 100+                         | --                               | ~10--100µm <br> (ROI-based)                   |
| DBiT-seq   | ~6,000-10,000           | ~30-100                      | --                               | ~10µm                                         |
| Spatial-CITE-seq | ~5,000-10,000           | ~100-200                     | --                               | ~20--50µm                                     |
| MALDI-MSI + IMC | --                      | ~30-50                       | 100s to <br> 1,000s              | ~1--10µm                                      |
| MALDI-MSI + Visium | 18000                   | --                           | 100-1000                         | ~10--50µm (MALDI); <br> 55µm (ST)             |

Altogether, spatial omics are accelerating insights into tissue biology
and transforming multiple areas of medicine. In particular, they have
been extensively applied to study complex microenvironments, such as
tumours, where spatial context and cellular heterogeneity are critical
and often obscured in bulk or single-modality data [17, 33-35].
They have also been used to study clonality differences in space [13,36].
As the applications are vast and rapidly evolving, they are reviewed in 
detail elsewhere [17,34,35].

With this foundation in the underlying technologies and their biological
potential, the next section shifts focus to the analytical challenges
and computational frameworks required to interpret and integrate spatial
omics data.

### References
[1] L. Liu, A. Chen, Y. Li, J. Mulder, H. Heyn, and X. Xu, "Spatiotemporal omics for biology and medicine," Cell, vol. 187, no. 17, pp. 4488--4519, 2024, doi: 10.1016/j.cell.2024.07.040.

[2] G. Palla, D. S. Fischer, A. Regev, and F. J. Theis, "Spatial components of molecular tissue biology," Nature Biotechnology, vol. 40, no. 3, pp. 308--318, 2022, doi: 10.1038/s41587-021-01182-1.

[3] E. Lundberg and G. H. H. Borner, "Spatial proteomics: A powerful discovery tool for cell biology," Nature Reviews Molecular Cell Biology, vol. 20, no. 5, pp. 285--302, 2019, doi: 10.1038/s41580-018-0094-y.

[4] S. Jing, H. Wang, P. Lin, J. Yuan, Z. Tang, and H. Li, "Quantifying and interpreting biologically meaningful spatial signatures within tumor microenvironments," npj Precision Oncology, vol. 9, no. 1, p. 68, 2025, doi: 10.1038/s41698-025-00857-1.

[5] J. W. Hickey et al., "Spatial mapping of protein composition and tissue organization: A primer for multiplexed antibody-based imaging," Nature Methods, vol. 19, no. 3, pp. 284--295, 2022, doi: 10.1038/s41592-021-01316-y.

[6] S. Magaki, S. A. Hojat, B. Wei, A. So, and W. H. Yong, "An Introduction to the Performance of Immunohistochemistry," in Biobanking, vol. 1897, W. H. Yong, Ed., New York, NY: Springer New York, 2019, pp. 289--298. doi: 10.1007/978-1-4939-8935-5_25.

[7] M. E. Ijsselsteijn et al., "Cancer immunophenotyping by seven‐colour multispectral imaging without tyramide signal amplification," The Journal of Pathology: Clinical Research, vol. 5, no. 1, pp. 3--11, 2019, doi: 10.1002/cjp2.113.

[8] J.-R. Lin et al., "Highly multiplexed immunofluorescence imaging of human tissues and tumors using t-CyCIF and conventional optical microscopes," eLife, vol. 7, p. e31657, 2018, doi: 10.7554/eLife.31657.

[9] A. J. Radtke et al., "IBEX: An iterative immunolabeling and chemical bleaching method for high-content imaging of diverse tissues," Nature Protocols, vol. 17, no. 2, pp. 378--401, 2022, doi: 10.1038/s41596-021-00644-9.

[10] Y. Goltsev et al., "Deep Profiling of Mouse Splenic Architecture with CODEX Multiplexed Imaging," Cell, vol. 174, no. 4, pp. 968--981.e15, 2018, doi: 10.1016/j.cell.2018.07.010.

[11] S. K. Saka et al., "Immuno-SABER enables highly multiplexed and amplified protein imaging in tissues," Nature Biotechnology, vol. 37, no. 9, pp. 1080--1090, 2019, doi: 10.1038/s41587-019-0207-y.

[12] S. Black et al., "CODEX multiplexed tissue imaging with DNA-conjugated antibodies," Nature protocols, vol. 16, no. 8, pp. 3802--3835, 2021.

[13] C. Giesen et al., "Highly multiplexed imaging of tumor tissues with subcellular resolution by mass cytometry," Nature Methods, vol. 11, no. 4, pp. 417--422, 2014, doi: 10.1038/nmeth.2869.

[14] L. Keren et al., "MIBI-TOF: A multiplexed imaging platform relates cellular phenotypes and tissue structure," Science Advances, vol. 5, no. 10, p. eaax5851, 2019, doi: 10.1126/sciadv.aax5851.

[15] M. Asp, J. Bergenstråhle, and J. Lundeberg, "Spatially Resolved Transcriptomes---Next Generation Tools for Tissue Exploration," BioEssays, vol. 42, no. 10, 2020, doi: 10.1002/bies.201900221.

[16] J. Du et al., "Advances in spatial transcriptomics and related data analysis strategies," Journal of Translational Medicine, vol. 21, no. 1, p. 330, 2023, doi: 10.1186/s12967-023-04150-2.

[17] L. Moses and L. Pachter, "Museum of spatial transcriptomics," Nature Methods, vol. 19, no. 5, pp. 534--546, 2022, doi: 10.1038/s41592-022-01409-2.

[18] K. H. Chen, A. N. Boettiger, J. R. Moffitt, S. Wang, and X. Zhuang, "Spatially resolved, highly multiplexed RNA profiling in single cells," Science, vol. 348, no. 6233, p. aaa6090, 2015, doi: 10.1126/science.aaa6090.

[19] C.-H. L. Eng et al., "Transcriptome-scale super-resolved imaging in tissues by RNA seqFISH+," Nature, vol. 568, no. 7751, pp. 235--239, 2019, doi: 10.1038/s41586-019-1049-y.

[20] A. Janesick et al., "High resolution mapping of the tumor microenvironment using integrated single-cell, spatial and in situ analysis," Nature Communications, vol. 14, no. 1, p. 8353, 2023, doi: 10.1038/s41467-023-43458-x.

[21] X. Wang et al., "Three-dimensional intact-tissue sequencing of single-cell transcriptional states," Science (New York, N.Y.), vol. 361, no. 6400, p. eaat5691, 2018, doi: 10.1126/science.aat5691.

[22] J. H. Lee et al., "Fluorescent in situ sequencing (FISSEQ) of RNA for gene expression profiling in intact cells and tissues," Nature Protocols, vol. 10, no. 3, pp. 442--458, 2015, doi: 10.1038/nprot.2014.191.

[23] P. L. Ståhl et al., "Visualization and analysis of gene expression in tissue sections by spatial transcriptomics," Science, vol. 353, no. 6294, pp. 78--82, 2016, doi: 10.1126/science.aaf2403.

[24] S. G. Rodriques et al., "Slide-seq: A scalable technology for measuring genome-wide expression at high spatial resolution," Science, vol. 363, no. 6434, pp. 1463--1467, 2019, doi: 10.1126/science.aaw1219.

[25] R. R. Stickels et al., "Highly sensitive spatial transcriptomics at near-cellular resolution with Slide-seqV2," Nature Biotechnology, vol. 39, no. 3, pp. 313--319, 2021, doi: 10.1038/s41587-020-0739-1.

[26] M. Aichler and A. Walch, "MALDI Imaging mass spectrometry: Current frontiers and perspectives in pathology research and practice," Laboratory Investigation, vol. 95, no. 4, pp. 422--431, 2015, doi: 10.1038/labinvest.2014.156.

[27] M. Stoeckli, P. Chaurand, D. E. Hallahan, and R. M. Caprioli, "Imaging mass spectrometry: A new technology for the analysis of protein expression in mammalian tissues," Nature Medicine, vol. 7, no. 4, pp. 493--496, 2001, doi: 10.1038/86573.

[28] C. R. Merritt et al., "Multiplex digital spatial profiling of proteins and RNA in fixed tissue," Nature Biotechnology, vol. 38, no. 5, pp. 586--599, 2020, doi: 10.1038/s41587-020-0472-9.

[29] Y. Liu et al., "High-Spatial-Resolution Multi-Omics Sequencing via Deterministic Barcoding in Tissue," Cell, vol. 183, no. 6, pp. 1665--1681.e18, 2020, doi: 10.1016/j.cell.2020.10.026.

[30] Y. Liu et al., "High-plex protein and whole transcriptome co-mapping at cellular resolution with spatial CITE-seq," Nature Biotechnology, vol. 41, no. 10, pp. 1405--1409, 2023, doi: 10.1038/s41587-023-01676-0.

[31] J. B. Nunes et al., "Integration of mass cytometry and mass spectrometry imaging for spatially resolved single-cell metabolic profiling," Nature Methods, vol. 21, no. 10, pp. 1796--1800, 2024, doi: 10.1038/s41592-024-02392-6.

[32] M. Vicari et al., "Spatial multimodal analysis of transcriptomes and metabolomes in tissues," Nature Biotechnology, vol. 42, no. 7, pp. 1046--1050, 2024, doi: 10.1038/s41587-023-01937-y.

[33] M. Abadi et al., "TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems." arXiv, 2016. doi: 10.48550/ARXIV.1603.04467.

[34] C. Engblom and J. Lundeberg, "Putting cancer immunotherapy into spatial context in the clinic," Nature Biotechnology, vol. 43, no. 4, pp. 471--476, 2025, doi: 10.1038/s41587-025-02596-x.

[35] W.-C. Hsieh et al., "Spatial multi-omics analyses of the tumor immune microenvironment," Journal of Biomedical Science, vol. 29, no. 1, p. 96, 2022, doi: 10.1186/s12929-022-00879-y.

[36] C. Engblom et al., "Spatial transcriptomics of B cell and T cell receptors reveals lymphocyte clonal dynamics," Science, vol. 382, no. 6675, p. eadf8486, 2023, doi: 10.1126/science.adf8486.
