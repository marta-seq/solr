
## Spatially Resolved Omics Techniques

Spatial omics technologies have the potential to advance our understanding 
of tumour ecosystems and improve clinical outcomes. 
The essence of spatial omics lies in its aptitude for the simultaneous detection 
of molecular constituents at exact spatial coordinates (35).
Spatial molecular profiling vary in resolution, scale and molecular
complexity (36).

### Spatial proteomics

Spatial proteomics encompasses technologies that enable the in-situ profiling
of proteins within tissues, preserving their spatial localization. 
Most spatial proteomics techniques detect proteins using antibodies tagged 
with fluorophores, metals, or DNA barcodes. These tags are then read using
technologies such as fluorescence microscopy, mass spectrometry, or DNA-based
imaging to map the spatial distribution of proteins within tissues (35, 37-39).
A summary of key features across these technologies is provided in Table1. 

Immunohistochemistry (IHC) is one of the most established clinical tools for protein
detection (40).  It uses enzyme-linked antibodies to generate a chromogenic signal 
visible under brightfield microscopy. While widely available and routinely used in
diagnostics, traditional IHC is generally limited to detecting one or a few markers per 
tissue section, making it unsuitable for high-dimensional spatial profiling.


Among the fluorescence-based approaches,  immunofluorescence (IF) (41) is widely 
used but limited in multiplexing. More advanced cyclic immunofluorescence methods—such
as tissue-based cyclic
immunofluorescence (t-CyCIF) \cite{B42_Cycif2018} and \gls{IBEX}\cite{B43_ibex_2022} - use iterative staining and imaging cycles to overcome spectral limitations, enabling the detection of 40–60+ markers with spatial resolution down to 200–300 nm.
Among the fluorescence-based approaches, immunofluorescence (IF)
\[CITATION\] is widely used but limited in multiplexing. More advanced
cyclic immunofluorescence methods---such as tissue-based cyclic
immunofluorescence (t-CyCIF) (40) and iterative bleaching extends
multiplexity (IBEX) (41) - use iterative staining and imaging cycles to
overcome spectral limitations, enabling the detection of 40--60+ markers
with spatial resolution down to 200--300 nm.

DNA-barcoded approaches such as co-detection by indexing (CODEX) (42)
and immunostaining with signal amplification by exchange reaction
(Immuno-SABER) further enhance multiplexing. CODEX utilizes DNA-barcoded
antibodies and sequential hybridization of fluorescent probes, achieving
high multiplexing (up to 60 proteins) with single-cell resolution (\~500
nm) in a single imaging plane (43). Immuno-SABER employs orthogonal DNA
concatemers for signal amplification ADD RESOLUTION AND PLEX (44).

Mass spectrometry-based approaches---notably Imaging Mass Cytometry
(IMC) (45) and multiplexed ion beam imaging by time of flight (MIBI-TOF)
(46)---use antibodies conjugated to isotopically pure lanthanide metals,
which are detected using laser ablation (IMC) or ion beams (MIBI). These
methods avoid fluorescence background and allow simultaneous
quantification of 40--50 proteins per tissue section. IMC offers spatial
resolution of approximately 1 μm, while MIBI achieves higher resolution
(\~300 nm), albeit with more complex instrumentation.

Table 1 -- Comparison table of key features of spatial proteomics
profiling methods

+--------+----------+--------------+---------+--------------+-------+
| Method | Tag type | Detection    | Multi   | Resolution   | Refe  |
|        |          |              | plexing |              | rence |
|        |          |              | c       |              |       |
|        |          |              | apacity |              |       |
+========+==========+==============+=========+==============+=======+
| IHC    | Enzyme   | Brightfield  | 1-2     |   ---------  |       |
|        | (ch      | Microscopy   |         |              |       |
|        | romogen) |              |         |   ---------  |       |
|        |          |              |         |              |       |
|        |          |              |         |   ---------  |       |
|        |          |              |         |   \~1--5 µm  |       |
|        |          |              |         |   ---------  |       |
|        |          |              |         |              |       |
|        |          |              |         |   ---------  |       |
+--------+----------+--------------+---------+--------------+-------+
| IF     | Flu      | Fluorescence | 4-7     | \~200--500   |       |
|        | orophore | Microscopy   |         | nm           |       |
+--------+----------+--------------+---------+--------------+-------+
| cyCIF  | Flu      | Cyclic       | \>60    | \~250 nm     |       |
| / 4i / | orophore | Fluorescence |         |              |       |
| IBEX   | (it      | Imaging      |         |              |       |
|        | erative) |              |         |              |       |
+--------+----------+--------------+---------+--------------+-------+
| CODEX  | DNA      | Fluorescence | \>      | \~250 nm     |       |
|        | Barcode  | Microscopy   | 40--60+ |              |       |
+--------+----------+--------------+---------+--------------+-------+
| Immuno | DNA      | Fluorescence | \>50+   | \~200--300   |       |
| -SABER | Barcode  | Microscopy   |         | nm           |       |
|        | (conc    |              |         |              |       |
|        | atemers) |              |         |              |       |
+--------+----------+--------------+---------+--------------+-------+
| IMC    | Metal    | Laser        | \~40    | \~1 µm       |       |
|        | Isotopes | Ablation +   |         |              |       |
|        |          | MS           |         |              |       |
+--------+----------+--------------+---------+--------------+-------+
| MIBI   | Metal    | Ion Beam +   | \~40    | \~300 nm     |       |
|        | Isotopes | MS           |         |              |       |
+--------+----------+--------------+---------+--------------+-------+

Together, these spatial proteomics platforms offer complementary
advantages in terms of marker throughput, resolution, and sensitivity,
enabling comprehensive characterization of the tumour microenvironment,
immune cell infiltration, and cellular architecture in situ.

**2.3.2 Spatial transcriptomics**

Spatial transcriptomics enables the study of gene expression within the
tissue architecture, preserving spatial context at cellular or
subcellular resolution. These platforms can be broadly divided into two
main categories based on detection strategy: imaging-based
methods---including *In Situ Hybridization* (ISH) and *In Situ
Sequencing* (ISS)---and spatial barcoding methods, which rely on
capture-based approaches followed by sequencing. Each approach presents
trade-offs in terms of resolution, transcriptome coverage, throughput,
and tissue compatibility (35,47--49) (Table 2).

*[In Situ Hybridization Imaging-Based Approaches]{.underline}*

ISH-based techniques use fluorescently labelled probes that hybridize
directly to target RNA molecules in fixed tissue sections. These methods
are highly accurate and can offer single-molecule and single-cell
resolution, but typically have limitations in multiplexing capacity
unless cyclic imaging or barcode strategies are used.

Basic single molecule FISH (*smFISH)* can detect individual transcripts
with high spatial precision but is limited in the number of genes
detectable due to the finite number of distinguishable fluorophores
\[citation\].

Advanced multiplexed methods, such as MERFISH, and seqFISH+, overcome
these limits by iterative cycles of hybridization and imaging, enabling
detection of hundreds to thousands of genes. Multiplexed error-robust
FISH (MERFISH) (50) uses combinatorial labelling and error-correcting
barcodes to detect thousands of RNA species in single cells. Sequential
fluorescence in situ hybridization (seqFISH+) (51) leverages sequential
rounds of hybridization with an expanded pseudo-color palette, enabling
detection of over 10,000 genes at subcellular resolution.

Commercial platforms such as CosMx (52), MERscope (Vizgen) \[citation\],
and Xenium (10x Genomics) (54) implement these strategies to achieve
high-resolution imaging of hundreds to thousands of RNA species, often
with optional co-detection of proteins.

*[In situ sequencing Imaging-Based Approaches]{.underline}*

ISS techniques sequence RNA molecules directly within tissues,
preserving both spatial context and nucleotide identity. Unlike ISH, ISS
provides sequence information, enabling mutation and splice isoform
detection.

STARmap (55) improves detection efficiency by using DNA nanostructures
and hydrogel-tissue chemistry, while FISSEQ (56), enables untargeted,
whole-transcriptome analysis through in situ reverse transcription and
random-hexamer priming. These methods retain spatial localization while
offering a more detailed molecular readout than hybridization alone.

Both ISH and ISS require fluorescence microscopy for imaging readouts
and are collectively referred to as imaging-based spatial
transcriptomics.

*[Spatial Barcoding and Sequencing-Based Methods]{.underline}*

Unlike imaging methods, spatial barcoding techniques rely on
sequencing-based detection. They use spatially encoded oligonucleotides
(barcodes) fixed to a surface (e.g., slide, bead, or grid) to capture
RNA from overlying tissue. After RNA capture, reverse transcription and
sequencing are performed, and spatial information is reconstructed based
on barcode identity.

Prominent examples include Visium (10X Genomics) (57), which captures
RNA on slide-mounted barcoded spots (\~55 µm resolution), and it's new
version, Visium HD \[citation\], that offers spatial resolution to
\~2--5 µm. Slide-seq (58) and Slide-seqV2 (59) use barcoded beads with
known spatial locations (\~10 µm resolution).

These methods offer broader transcriptomic coverage---often approaching
the whole transcriptome---and are scalable to larger tissue sections.
However, they typically offer lower spatial resolution than
imaging-based platforms and may not reach single-cell or subcellular
detail.

.

Table 2 - Comparison table of key features of spatial transcriptomics
profiling methods

+---------------+------------+----------------+------------+---------+
| Method        | Category   | Resolution     | Transcript | Re      |
|               |            |                | coverage   | ference |
+===============+============+================+============+=========+
| smFISH        | ISH        | \~200--300 nm  | Low (1--10 |         |
|               | (Imag      | (subcellular)  | genes)     |         |
|               | ing-based) |                |            |         |
+---------------+------------+----------------+------------+---------+
| MERFISH       | ISH        | \~100--300 nm  | High       |         |
|               | (Imag      |                | (1,000+    |         |
|               | ing-based) | (subcellular)  | genes)     |         |
+---------------+------------+----------------+------------+---------+
| seqFISH /     | ISH        | \~200 nm       | Very High  |         |
| seqFISH+      | (Imag      |                | (\>10,000  |         |
|               | ing-based) | (subcellular)  | genes)     |         |
+---------------+------------+----------------+------------+---------+
| CosMx         | ISH /      | \~250 nm       | Up to      |         |
| (NanoString)  | smFISH     | (subcellular)  | \~1,000    |         |
|               |            |                | genes      |         |
+---------------+------------+----------------+------------+---------+
| MERscope      | MER        | \~300 nm       | \~         |         |
| (Vizgen)      | FISH-based |                | 500--1,000 |         |
|               | (ISH)      | (subcellular)  | genes      |         |
+---------------+------------+----------------+------------+---------+
| Xenium (10x   | sm         | \~280 nm       | \~300--400 |         |
| Genomics)     | FISH-based | (subcellular)  | genes      |         |
|               | (ISH)      |                |            |         |
+---------------+------------+----------------+------------+---------+
| STARmap       | ISS        | \~2--3 µm (3D) | \~1,000    |         |
|               |            |                | genes      |         |
|               |            | (Single-cell)  |            |         |
+---------------+------------+----------------+------------+---------+
| FISSEQ        | ISS (In    | \~300 nm -- 1  | Whole      |         |
|               | Situ       | µm             | tra        |         |
|               | S          |                | nscriptome |         |
|               | equencing) | (Single-cell/  |            |         |
|               |            | subcellular)   |            |         |
+---------------+------------+----------------+------------+---------+
| Visium (10x   | Spatial    | \~55 µm (spot  | Whole      |         |
| Genomics)     | Barcoding  | diameter)      | tra        |         |
|               | (Slide)    |                | nscriptome |         |
+---------------+------------+----------------+------------+---------+
| Visium HD     | Spatial    | \~2--5 µm      | Whole      |         |
|               | Barcoding  |                | tra        |         |
|               | (Slide)    | (close         | nscriptome |         |
|               |            | single-cell)   |            |         |
+---------------+------------+----------------+------------+---------+
| Slide-seq     | Spatial    | \~10 µm (bead  | Whole      |         |
|               | Barcoding  | diameter)      | tra        |         |
|               | (Beads)    |                | nscriptome |         |
+---------------+------------+----------------+------------+---------+
| Slide-seqV2   | Spatial    | \~10 µm        | Whole      |         |
|               | Barcoding  | (higher        | tra        |         |
|               | (Beads)    | sensitivity)   | nscriptome |         |
+---------------+------------+----------------+------------+---------+

**2.3.3 Spatial metabolomics**

Spatial metabolomics explores the spatial distribution of metabolites
directly in tissue sections, providing insight into biochemical activity
within the anatomical context (60). The field is largely driven by
Matrix-Assisted Laser Desorption/Ionization Mass Spectrometry Imaging
(MALDI-MSI) (61), a label-free, untargeted technique capable of
detecting a wide range of small molecules including lipids,
neurotransmitters, and drugs.

In MALDI-MSI, tissue sections are coated with a matrix that facilitates
ionization when hit by a laser. The resulting ions are analysed by mass
spectrometry to reconstruct spatial metabolite maps. MALDI-MSI offers
spatial resolution in the range of 10--50 µm, with coverage of hundreds
to thousands of metabolites, depending on the tissue and matrix.

**2.3.4 Spatial multi-omics**

Spatial multi-omics technologies (Table 3) enable the simultaneous or
integrative profiling of multiple molecular layers---such as RNA,
proteins, and metabolites---within their spatial tissue context,
offering a more comprehensive understanding of cellular states and
interactions.

One of the most established platforms, GeoMx Digital Spatial Profiler
(DSP) (NanoString) (62), allows for high-plex profiling of both RNA and
proteins within defined regions of interest using oligonucleotide-tagged
probes and UV-directed barcode collection, though at limited spatial
resolution (\~10--100 μm). Other advanced methods, such as deterministic
barcoding in tissue for spatial omics sequencing (DBiT-seq) (63),
achieve co-detection of RNA and proteins through microfluidic-based
spatial barcoding on the same section, offering high spatial resolution
(\~10 μm) and true multimodal readouts. Similarly, Spatial-CITE-seq (64)
adapts co-indexing of transcriptomes and epitopes (CITE) to the spatial
dimension, enabling the capture of transcriptomes alongside surface
protein markers.

Furthermore, MALDI-MSI have been successfully combined on the same slide
with both IMC (65) and spatial transcriptomics platforms like 10x Visium
(66).

Table 3 - Comparison table of key features of spatial multiomics methods

+---------+----------+--------------+---------+--------------+-------+
| Method  | Spatial  | RNA Targets  | Protein | Metabolite   | Refe  |
|         | Re       |              | Targets | Coverage     | rence |
|         | solution |              |         |              |       |
+=========+==========+==============+=========+==============+=======+
| GeoMx   | \~10     | Up to        | \~100+  | --           |       |
| DSP     | --100 µm | \~18,000     | (a      |              |       |
| (Nano   | (RO      | (targeted    | ntibody |              |       |
| String) | I-based) | panels)      | panels) |              |       |
+---------+----------+--------------+---------+--------------+-------+
| D       |   -      | \~6          | \~      | --           |       |
| BiT-seq | -------- | ,000--10,000 | 30--100 |              |       |
|         |          |              |         |              |       |
|         |  \~10 µm |              |         |              |       |
|         |   -      |              |         |              |       |
|         | -------- |              |         |              |       |
|         |          |              |         |              |       |
|         |   -      |              |         |              |       |
|         | -------- |              |         |              |       |
|         |          |              |         |              |       |
|         |   -----  |              |         |              |       |
|         |          |              |         |              |       |
|         |   -----  |              |         |              |       |
+---------+----------+--------------+---------+--------------+-------+
| Sp      | \~2      | \~5          | \~1     | --           |       |
| atial-C | 0--50 µm | ,000--10,000 | 00--200 |              |       |
| ITE-seq |          |              |         |              |       |
+---------+----------+--------------+---------+--------------+-------+
| MALD    | \~       | --           | \       | 100s to      |       |
| I-MSI + | 1--10 µm |              | ~30--50 | 1,000s       |       |
| IMC     |          |              |         |              |       |
+---------+----------+--------------+---------+--------------+-------+
| MALD    | \~1      | \~18,000     | --      | 100s to      |       |
| I-MSI + | 0--50 µm |              |         | 1,000s       |       |
| Visium  | (MALDI); |              |         |              |       |
|         | 55 µm    |              |         |              |       |
|         | (ST)     |              |         |              |       |
+---------+----------+--------------+---------+--------------+-------+

While this overview focuses on representative technologies, detailed
descriptions and additional methods and recent innovations are
extensively reviewed elsewhere.
