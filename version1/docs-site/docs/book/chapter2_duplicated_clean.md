---
bibliography: bibliography.bib
csl: ieee.csl
---

# Analytical Frameworks for Spatial Omics Data

The increasing availability of high-resolution spatial omics
data-particularly at subcellular resolution-brings new computational
challenges, including those related to the spatial dimension, increased
data complexity, and multiplexing. This chapter focuses on the analysis
of single-molecule spatial transcriptomics and proteomics, where each
detected molecule is mapped with precise spatial coordinates. While the
field is rapidly evolving, with a growing number of specialized tools
and packages emerging at a fast pace, this work will centre on the core
analytical pipeline common to most imaging-based platforms. Methods
specific to certain technologies-such as deconvolution approaches
designed for lower-resolution platforms like 10x Visium-or complex
multimodal integration strategies with single-cell data will not be
covered here.

## Data preprocessing

Both spatial proteomics and spatial transcriptomics require dedicated
preprocessing steps to mitigate technical noise and artifacts that may
obscure true biological signals. These preprocessing strategies differ
significantly, reflecting the distinct nature of the underlying
technologies.

### Spatial proteomics

For spatial proteomics noise sources can vary based on the antibodies,
detection channels, and tissue types used, and may manifest as artifacts
such as hot pixels, shot noise, background noise and channel crosstalk
[1-3] and may also exhibit low signal-to-noise ratios [4].

Channel crosstalk, where signals from one channel interfere with
adjacent channels, can be mitigated through well-designed antibody
panels [4] or correction methods like CATALYST [3], which uses
pre-acquisition compensation matrices, as well as post-processing
techniques [2,5].

Beyond crosstalk, other sources of noise - such as hot pixels,
background signal, and shot noise - require distinct computational
strategies. Some studies have employed adjacent approaches, including
those implemented in Steinbock [6] and MAUI (MBI analysis interface)
[2], while others have developed "homebrew" methods based on
traditional image filtering techniques [5, 7,8]. However,
these methods are often insufficient for fully addressing complex noise
patterns.

Tools like Ilastik [9] provide supervised pixel classification to
distinguish background noise from true signal on a per-marker basis
[6,9,10]. Although this approach requires extensive manual
annotation, it is currently considered state-of-the-art, as it
effectively removes background noise and improves signal normalization
and batch effect reduction across samples [7]. More recently,
IMC-Denoise [11] has been introduced as a two-step pipeline combining
traditional algorithms with a self-supervised deep learning model based
on Noise2Void [12].

### Spatial transcriptomics

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

## Cell segmentation

A central challenge in spatial omics is the accurate assignment of
molecular measurements-transcripts or proteins-to individual cells.
This process, known as cell segmentation, is essential for translating
spatial data into single-cell level insights and directly influences
downstream biological interpretation.

### Spatial proteomics

In spatial proteomics, cell segmentation is typically performed using
the imaging data generated during acquisition. These multiplexed images
often include nuclear stains to identify nuclei and membrane markers to
delineate cell boundaries. These signals are combined into RGB or
multi-channel images and processed using image analysis pipelines.

Traditional image segmentation approaches, such as thresholding, edge
detection, and the watershed algorithm, are commonly employed. These
methods are implemented in established tools like CellProfiler [13],
[14] and ImageJ/Fiji [15], which allow customization of segmentation
pipelines using scripting interfaces. In recent years, DL models have
gained prominence due to their improved accuracy, generalizability, and
community availability [16]. Notably, Mesmer, part of the DeepCell
framework, DL-based model specifically trained on spatial proteomics
datasets, leveraging TissueNet, a large annotated dataset [17].
Another widely adopted method is Cellpose, a generalist DL segmentation
tool that predicts vector flows to delineate cell boundaries instead of
relying on direct pixel-wise classification. Though originally trained
on a broad array of microscopy images, Cellpose also offers specialist
models, including those fine-tuned on TissueNet subsets to enhance
performance on spatial proteomics images [18,19].

Segmentation strategies in spatial proteomics range from nucleus-based
expansion methods, which approximate cell outlines from DAPI staining,
to full-cell segmentation techniques that incorporate both nuclear and
membrane markers. A growing number of domain-specific models continue to
emerge, each tailored to address the distinct noise profiles, spatial
resolution, and multiplexing characteristics of spatial proteomics data
[20-24].

### Spatial transcriptomics

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
dependency on auxiliary images [25-33].

A prominent example is Baysor [25], a general probabilistic framework
implemented in Julia. Based on Markov Random Field (MRF), Baysor models
cells by jointly considering spatial proximity and transcriptional similarity. It can
incorporate prior segmentations, assigning confidence scores to them, or
operate de novo using only transcript data. Baysor functions in both 2D
and 3D, optimizing the likelihood that neighbouring transcripts
originate from the same cell. Notably, it has shown strong performance
across datasets and has been reported to outperform the default
segmentation provided in Xenium workflows [34], highlighting its
practical utility and growing adoption.

## Cell phenotyping

Cell phenotyping is the process of assigning biological identities to
individual cells based on their molecular profiles, such as gene or
protein expression, and serves as a critical step in interpreting
spatial omics data.

### Spatial proteomics

Unlike transcriptomic approaches, spatial proteomics typically profiles
a pre-selected panel of ~30-50 protein markers,
chosen based on the biological question. This reduced and curated
feature space simplifies downstream computational analysis but increases
reliance on prior biological knowledge for interpreting clusters [6].

Commonly used clustering algorithms include Cytosplore, a GUI which
leverages Hierarchical Stochastic Neighbour Embedding (HSNE) [35],
Phenograph that builds a shared nearest neighbor graph and uses Louvain 
algorithm to partition of the graph into communities [35], and FlowSOM that applies
Self Organizing maps [36,37].

In addition to unsupervised clustering, other strategies for phenotyping
include: Manual gating, where marker thresholds are used to define known
populations based on expert knowledge; Supervised machine learning
models, such as Random Forests, which classify cells using labelled
training data and reference mapping, where new datasets are aligned to
previously annotated references to infer phenotypes.

### Spatial transcriptomics

Spatial transcriptomics datasets that approach single-cell resolution
often adopt analysis pipelines developed for single-cell RNA (scRNA). This
involves several key steps:

The process begins with data preprocessing, where cells of low quality
are removed through quality control steps. Following this, normalization
is applied-most commonly scaling gene counts to 10,000 per cell and
applying a log-transformation-to standardize data across cells. Since
these datasets are high-dimensional, dimensionality reduction is used to
make the data more manageable. Principal Component analysis (PCA) is commonly
applied first to identify the most informative gene expression patterns.
For visualization and further analysis, nonlinear techniques like t-SNE and 
UMAP are used to project the data into two or three dimensions while preserving
its underlying structure [38].

Next, clustering algorithms group similar cells together. This is
usually done by building a KNN graph in the PCA-reduced space. Each cell
is connected to its K most similar neighbours, and clustering algorithms
such as Louvain [39] and Leiden [40], often with better performance
[41] are applied to identify densely connected communities within the
graph. These clusters are assumed to represent groups of cells with
similar transcriptional profiles, often corresponding to distinct cell
types or states.

Once clusters are identified, cell type annotation is performed to
assign biological identities to the groups of cells. This can be done
either manually or automatically.

Manual annotation relies on known marker genes-genes that are
characteristically expressed in specific cell types. In practice, this
can be done by checking whether known markers are expressed in each
cluster or, conversely, by identifying differentially expressed genes
within clusters and matching them to known biological signatures. While
manual annotation is transparent and interpretable, it can be
subjective, labour-intensive, and limited by the availability and
specificity of known markers [38].

Alternatively, automated annotation methods are also available, which
leverage reference gene atlases and machine/deep learning models (e.g.,
CellTypist [42]), or label transfer approaches (e.g., Symphony
[43]). These methods provide faster and more scalable annotation but
tend to be less interpretable and dependent on the similarity between
the reference and the query data [38].

Currently, automated approaches are not yet standard for spatial
transcriptomics, particularly in cases where the data is limited by
lower gene counts, targeted panels, or lower sequencing depth compared
to single-cell RNA. Additionally, many spatial datasets do not align
well with existing reference atlases due to differences in tissue
context, resolution, or platform. As a result, manual annotation remains
more reliable and commonly used in spatial transcriptomics, particularly
when high-resolution spatial context and known marker expression are
available.

## Cellular neighbourhood analysis

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
differences in the underlying molecular readouts [6,38].

Once a graph is established, it enables the exploration of spatial
interactions between annotated cell types. A common approach involves
the computation of neighbourhood enrichment scores, which statistically
test whether specific cell types are found adjacent to each other more
or less frequently than expected by chance. These analyses rely on
permutation-based null models, which shuffle cell-type labels while
preserving tissue structure, providing a robust statistical framework to
assess enrichment or depletion of interactions [44]. Another related
method is the computation of co-occurrence scores, which estimate how
likely it is to observe specific cell-type pairs within increasing radii
around each cell. These scores reflect the conditional probability of
observing a certain cell type given the presence of another nearby,
offering an interpretable measure of spatial association across scales
[38].

Alternatively, simple interaction matrices can be computed, summarizing
the raw counts of neighbouring cell-type pairs [6]. Though not
statistical tests, these matrices are useful for exploratory data
analysis.

Beyond pairwise interaction analyses, more integrative approaches aim to
cluster cells based on the composition of their local neighbourhoods.
One strategy involves computing, for each cell, the fraction of
surrounding cell types within its local environment (e.g., 20 nearest
neighbours). These local composition profiles can then be clustered
using unsupervised methods such as k-means or Leiden algorithms,
grouping cells into recurring spatial neighbourhoods. This approach was
introduced by studies such as Goltsev et al. [45] and Schürch et al.
[46], which revealed that specific combinations of neighbouring
discrete cell types are often spatially organized in conserved patterns.

A related method involves aggregating gene or protein expression
features across the neighbourhood, effectively capturing a summary of
the local microenvironment. These aggregated features can then serve as
the basis for clustering or downstream modelling, this is approach is
implemented for example in covariance environment (COVET) [47],
NicheCompass, a graph deep-learning approach to identify and
quantitatively characterize niches by learning cell embeddings encoding
signalling events as spatial gene program activities [48] and Banksy
[49].

Through these computational strategies, spatial neighbourhoods can be
identified and interpreted biologically as spatial niches-localized,
recurring configurations of cells and their molecular environment that
reflect functionally relevant tissue microenvironments. Spatial niches
are particularly informative in contexts such as immuno-oncology, where
immune-tumour interactions shape the progression or suppression of
disease.

## Spatial domains

Closely related to cellular neighbourhoods, spatial domains usually
refer to broader, tissue-scale regions characterized by coherent gene or
protein expression patterns and underlying structural organization
[38]. While cellular neighbourhoods capture the immediate
microenvironment around a cell-defined by local interactions and
spatial proximity-spatial domains extend this concept to larger
anatomical or functional areas of tissue. These regions often consist of
diverse cell types and multiple neighbourhood configurations, working
together to support complex biological functions.

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
to be assigned the same label [50,51]. This assumes that a cell's
transcriptome resembles the average of its domain, although this may not
always hold true in tissues where multiple cell types are intermixed.
Other approaches employ deep learning, particularly graph-based neural
networks that can incorporate either histological information [52,53]
or spatial graphs derived purely from molecular and positional
data [54-56]. These methods offer flexibility and can
capture complex spatial dependencies, but their performance can vary
with dataset size and structure.

More recently, methods introducing hierarchical and multiscale
representations of spatial structure have emerged. For instance, NeST
[57] identifies nested co-expression hotspots-spatially contiguous
regions that co-express subsets of genes-by simultaneously searching
across gene and spatial dimensions. This approach captures biological
structure at multiple scales and can reveal overlapping or nested
domains, without requiring prior assumptions about gene sets or spatial
resolution. Similarly, the concept of tissue schematics [58] has been
proposed as a way to abstract spatial organization into higher-order
motifs, by identifying modular assemblies of cellular neighbourhoods.
These schematics can be used to represent tissue architecture in both
healthy and diseased states, offering insights into how local
interactions scale up to tissue-level functionality.

## Spatially Variable genes

Spatially Variable genes (SVG)s are genes whose expression displays
significant spatial patterning across a tissue. Identifying SVGs reveals
insights into tissue architecture, cell-cell communication, and local
microenvironmental influences, extending beyond what is captured by cell types
or spatial domains alone. These spatial patterns may arise from gradients in signalling
molecules, regional functional differences, or heterogeneous cell
compositions [38]. Notably, SVG detection does not rely on cell
segmentation, making it applicable across different spatial omics
technologies.

Several computational methods have been developed to detect SVGs by
decomposing gene expression variability into spatial and non-spatial
components. A widely used statistic is Moran's I, which quantifies
spatial autocorrelation by measuring how similar gene expression is
between neighbouring spots [59]. Model-based approaches like SpatialDE
[60], trendseek [61], SPARK [62], and SPARK-X [63] use Gaussian
processes or spatial statistical models to test for spatially structured
expression while accounting for noise and overdispersion. While these
tools test each gene independently and return p-values, they often
overlook spatial domain context, which can limit biological
interpretability. Other strategies take different modelling
perspectives: Sepal [64] applies a Gaussian diffusion process, scGCO
[65] uses graph cuts to detect spatial expression boundaries, and
SpaGCN [53], defines both domains and SVGs integrating histology and
spatial proximity through graph convolutional networks.

Importantly, the identification of SVGs is modality-agnostic and
applicable to both spatial transcriptomics and spatial proteomics.
Regardless of whether gene expression or protein abundance is measured,
spatially variable features represent key molecular signatures of tissue
structure and function, and form an essential layer of spatial omics
analysis.

## Functional analysis

Understanding how spatial context shapes cellular function is an
increasingly important focus in spatial omics. Beyond identifying cell
types, neighbourhoods, or spatially variable genes, functional analysis
aims to reveal how cells operate and interact in situ. This includes
investigating intracellular signalling pathways, transcription factor (TF)
activity, and particularly cell–cell communication (CCC) events, often
inferred from curated interaction networks derived from transcriptomics
data [66,67].

A broad landscape of computational tools has emerged for CCC analysis
[67-76], reflecting the growing complexity and richness of spatial omics
data. Broadly, these methods fall into two sections: identifying pairs
of genes that interact, such that expression of the gene in one cell
influences that of the other gene in others; and identifying pairs of
cells in which that gene pair interacts [77].

Most methods for CCC leverage ligand–receptor (L-R) interactions,
relying heavily on prior biological knowledge, often curated into 
databases such as CellPhoneDB [69], OmniPath [78], and CellChat [70].
These databases catalogue known signalling pathways and interaction pairs from the literature.
Using such resources, CCC inference can be extended to the spatial
context by identifying L-R pairs co-expressed in nearby cells [67]
for example using spatial co-expression [67, 73], or incorporating
different approaches such as optimal transport frameworks [72] or
Bayesian multiple instance learning (MIL) [76]
to more precisely link cells by both location and expression.

Complementing these prior knowledge-driven approaches are data-driven
methods that model spatial gene expression to discover novel
interactions. These models, like NCEM [68], GCNG [75] or SVCA [79]
and Misty [80] (while not specific for CCC), aim to capture
interactions that explain spatial expression variance across multiple
genes. They enable inference of new communication patterns not captured
in curated L-R databases, though their accuracy depends on model design
and training data.

Furthermore, some CCC frameworks estimate global co-localization across
entire tissue sections [68, 71, 80], and others focus on local
cell-cell proximity [72,73].

In parallel, there is increasing interest in inferring intracellular
functional activity; such as pathway activation, TF activity, or gene
set enrichment; within individual cells or regions. Tools like
decouplR [81] apply prior knowledge from pathway perturbation
signatures [82] or regulatory networks [83] to estimate these
functional states at single-cell or spot level. As with CCC, these
functional scores can then be spatially mapped. These approaches are
currently more mature for spatial transcriptomics, where broader gene
coverage enables more reliable functional inference. In contrast,
spatial proteomics remains limited by smaller marker panels and less
comprehensive prior knowledge, constraining the resolution of functional
state estimation.

While spatial omics provides a key advantage over single-cell RNA
through direct measurement of cell proximity-eliminating the need for
probabilistic neighbourhood modelling and enabling exclusion of
implausible, long-range interactions-this often comes with trade-offs.
Platforms with true cellular resolution (e.g., CosMx, Xenium) may
capture fewer genes, complicating pathway-level analyses and downstream
functional interpretation. Moreover, the heavy reliance on curated
databases across most CCC methods introduces variability and potential
inconsistencies, as predictions can differ significantly depending on
the resource used.

Ultimately, understanding how a cell's functional state is shaped by its
environment; and how it, in turn, influences surrounding cells; is a
central and still largely untapped frontier in spatial biology. As
methods advance, this bidirectional view of cellular function in context
promises to transform our understanding of tissue organization, disease
progression, and therapeutic targeting.

## How to distinguish between sample conditions

While the tasks described in the previous sections---such as
preprocessing, cell segmentation, and feature extraction---lay the
foundation for spatial omics analysis, deriving meaningful biological
insights often requires comparing groups, such as different disease
states, treatment responses, or tissue types.

### Summary statistics

One of the earliest approaches involves aggregating spatial features
across samples and comparing them using statistical summaries. For
example, scores for cell--cell communication, number of neighbours, or
other spatial statistics can be averaged per sample and used to detect
differences between conditions [67, 80,84]. However, this
strategy is often too simplistic, as it can blur important heterogeneity
by averaging out spatially localized effects.

### Matrix factorization methods

Matrix factorization methods provide a more nuanced alternative. These
approaches reduce high-dimensional data into a smaller set of latent
factors that capture major patterns of variation. Classical approaches
like PCA achieve this through linear decompositions, but newer tools
like ulti-Omics Factor Analysis (MOFA) [85 - 87], MEFISTO [88], and
DIALOGUE [89] extend the concept to handle multiple data types, 
structured variation, or intercellular coordination.
These methods can incorporate spatial information embedded in the matrix.

MOFA and its variants allow joint analysis of multi-omic datasets,
identifying latent factors that capture both shared and group-specific
variation. Other tools like DIALOGUE [89] and scITD [90] focus on
identifying coordinated gene programs across multiple cell types,
enabling the characterization of multicellular processes in an
unsupervised fashion.

NMF provides a more interpretable, parts-based representation, especially useful for
identifying additive biological signals like distinct cell states or
spatial domains. Spatially-aware versions of NMF, such as
Nonnegative Spatial Factorization (NSF), extend this
idea by directly incorporating spatial coordinates, better capturing
localized gene expression patterns [91].

These matrix-based methods not only enhance the ability to uncover
biologically relevant spatial and intercellular patterns, but also
enable unsupervised identification of group-level differences-whether
driven by disease, environment, or treatment - without requiring
explicit labels.

### Machine Learning and Deep Learning approaches

Beyond matrix factorization, supervised Machine Learning (ML) and
Deep Learning (DL) methods are increasingly being explored
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
spatial omics pipeline, such as image preprocessing, segmentation,
domain identification, phenotyping, and modelling of cell-cell
communication. But its use for direct classification or outcome
prediction is still sparse.

Among early efforts in this direction, NaroNet [92] introduced a patch
contrastive learning strategy combined with graph representations to
predict patient outcomes from IMC data. In the same cohort, Fu et
al.[93] proposed a deep multimodal graph-based network that integrates
IMC data with clinical variables to predict cancer survival. In another
example, Risom et al.[94] applied a random forest classifier to a MIBI
dataset of ductal carcinoma in situ (DCIS), using over 400 spatial
features-including tissue compartment enrichment and TME
morphometrics-to distinguish between progressors and non-progressors.

A more recent and notable advance is S3-CIMA [95], a weakly
supervised, single-layer convolutional neural network that learns
disease-specific spatial compositions of the tumour microenvironment. By
modelling local cell-type organizations from high-dimensional proteomic
imaging data (IMC and CODEX), it enables the discovery of
outcome-associated microenvironments in colorectal cancer. Similarly,
graph deep learning has been used to predict prognosis in gastric
cancer, where Cell-Graphs built from multiplexed immunohistochemistry
(mIHC) data enable prognosis prediction from spatial arrangements of
cell types [96].

Important to note is that DL methods are already well-established in
computational pathology and digital histopathology, where large
annotated datasets and well-defined visual features have allowed CNNs to
thrive in image classification, segmentation, and prognosis prediction
tasks [97, 98].

## Frameworks and tools for spatial omics analysis

A growing ecosystem of software tools supports the analysis of
single-cell and spatial omics data. Seurat (R) [99] and Scanpy
(Python) [100] are widely used for single-cell analysis. Scanpy is
built around the efficient AnnData structure [101], while Seurat uses
its own SeuratObject. Dedicated spatial omics tools such as Giotto (R)
[102] and Squidpy (Python) [84] offer integrated workflows for
spatial statistics, neighbourhood analysis, and visualization with also
new data frameworks like SpatialData (Python) [103] emerging to better
support spatial modalities. Napari [104], a general-purpose image
viewer, complements these tools with interactive, high-dimensional image
visualization, useful for working with spatial coordinates and tissue
images.

Python and R remain the dominant programming languages in this space.
Python is increasingly favoured for spatial omics due to its speed,
memory efficiency, and compatibility with deep learning and image
processing (e.g., PyTorch), while R remains popular for exploratory
analysis and visualization [105]. Meanwhile, Julia is gaining attention
for its performance, with tools like Baysor [25] highlighting its
potential in spatial omics analysis [106].

Given the rapid growth of tools - and the risk of incompatibilities in
data formats, APIs, and user interfaces - standardization efforts like
scverse [107] have emerged. These initiatives promote well-maintained,
interoperable core functionality, supporting a more cohesive and
collaborative spatial omics software ecosystem.

## Closing remarks

This review has focused on spatial omics methods that operate on the
basis of cell-defined units. However, it is important to note that many
of these approaches can also be applied at the transcript level,
bypassing the need for explicit cell segmentation. 
This is particularly true for tasks such as domain identification.
While cell-based analysis remains the prevailing standard in
spatial omics, alternative strategies that complement or replace cell
segmentation are gaining ground [56, 108]. 
This shift is partly driven by the technical difficulty of accurately
segmenting individual cells, especially in complex tissues
where cells may be overlapping, densely packed, or poorly defined.

Furthermore, this review has specifically addressed computational
methods applicable to single-cell resolution spatial data, and does not
cover broader-scale approaches such as spatial deconvolution or data
imputation, which are more relevant to spot-based or lower-resolution
platforms.

Even within the focused scope of single-cell approaches, many valuable
tools and developments could not be fully covered, reflecting the sheer
volume and velocity of innovation in the field. 


The increasing availability of spatially resolved data across multiple
modalities has fuelled an avalanche of computational methods, making it
increasingly difficult for researchers to keep pace and make informed
decisions about which tools best suit their needs. In this evolving
landscape, there is a growing need for "living reviews" [105],
curated repositories, and community-driven benchmarking [98] efforts
that can adapt to the field's rapid progress. This is the main motivation for
this repository. 

### References

[1] H. Baharlou, N. P. Canete, A. L. Cunningham, A. N. Harman, and E. Patrick, "Mass Cytometry Imaging for the Study of Human Diseases---Applications and Data Analysis Strategies," Frontiers in Immunology, vol. 10, p. 2657, 2019, doi: 10.3389/fimmu.2019.02657.

[2] A. Baranski et al., "MAUI (MBI Analysis User Interface)---An image processing pipeline for Multiplexed Mass Based Imaging," PLOS Computational Biology, vol. 17, no. 4, p. e1008887, 2021, doi: 10.1371/journal.pcbi.1008887.

[3] S. Chevrier, H. L. Crowell, V. R. T. Zanotelli, S. Engler, M. D. Robinson, and B. Bodenmiller, "Compensation of Signal Spillover in Suspension and Imaging Mass Cytometry," Cell Systems, vol. 6, no. 5, pp. 612--620.e5, 2018, doi: 10.1016/j.cels.2018.02.010.

[4] V. Milosevic, "Different approaches to Imaging Mass Cytometry data analysis," Bioinformatics Advances, vol. 3, no. 1, p. vbad046, 2023, doi: 10.1093/bioadv/vbad046.

[5] Y. J. Wang et al., "Multiplexed In Situ Imaging Mass Cytometry Analysis of the Human Endocrine Pancreas and Immune System in Type 1 Diabetes," Cell Metabolism, vol. 29, no. 3, pp. 769--783.e4, 2019, doi: 10.1016/j.cmet.2019.01.003.

[6] J. Windhager et al., "An end-to-end workflow for multiplexed image processing and analysis," Nature Protocols, vol. 18, no. 11, pp. 3565--3613, 2023, doi: 10.1038/s41596-023-00881-0.

[7] L. Keren et al., "A Structured Tumor-Immune Microenvironment in Triple Negative Breast Cancer Revealed by Multiplexed Ion Beam Imaging," Cell, vol. 174, no. 6, pp. 1373--1387.e19, 2018, doi: 10.1016/j.cell.2018.08.039.

[8] A. F. Rendeiro et al., "The spatial landscape of lung pathology during COVID-19 progression," Nature, vol. 593, no. 7860, pp. 564--569, 2021, doi: 10.1038/s41586-021-03475-6.

[9] S. Berg et al., "Ilastik: Interactive machine learning for (bio)image analysis," Nature Methods, vol. 16, no. 12, pp. 1226--1232, 2019, doi: 10.1038/s41592-019-0582-9.

[10] M. E. Ijsselsteijn, A. Somarakis, B. P. F. Lelieveldt, T. Höllt, and N. F. C. C. De Miranda, "Semi‐automated background removal limits data loss and normalizes imaging mass cytometry data," Cytometry Part A, vol. 99, no. 12, pp. 1187--1197, 2021, doi: 10.1002/cyto.a.24480.

[11] P. Lu et al., "IMC-Denoise: A content aware denoising pipeline to enhance Imaging Mass Cytometry," Nature Communications, vol. 14, no. 1, p. 1601, 2023, doi: 10.1038/s41467-023-37123-6.

[12] A. Krull, T.-O. Buchholz, and F. Jug, "Noise2Void - Learning Denoising From Single Noisy Images," in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Long Beach, CA, USA: IEEE, 2019, pp. 2124--2132. doi: 10.1109/CVPR.2019.00223.

[13] C. McQuin et al., "CellProfiler 3.0: Next-generation image processing for biology," PLOS Biology, vol. 16, no. 7, p. e2005970, 2018, doi: 10.1371/journal.pbio.2005970.

[14] A. E. Carpenter et al., "CellProfiler: Image analysis software for identifying and quantifying cell phenotypes," Genome Biology, vol. 7, no. 10, p. R100, 2006, doi: 10.1186/gb-2006-7-10-r100.

[15] J. Schindelin et al., "Fiji: An open-source platform for biological-image analysis," Nature Methods, vol. 9, no. 7, pp. 676--682, 2012, doi: 10.1038/nmeth.2019.

[16] L. Von Chamier et al., "Democratising deep learning for microscopy with ZeroCostDL4Mic," Nature Communications, vol. 12, no. 1, p. 2276, 2021, doi: 10.1038/s41467-021-22518-0.

[17] N. F. Greenwald et al., "Whole-cell segmentation of tissue images with human-level performance using large-scale data annotation and deep learning," Nature Biotechnology, vol. 40, no. 4, pp. 555--565, 2022, doi: 10.1038/s41587-021-01094-0.

[18] C. Stringer, T. Wang, M. Michaelos, and M. Pachitariu, "Cellpose: A generalist algorithm for cellular segmentation," Nature Methods, vol. 18, no. 1, pp. 100--106, 2021, doi: 10.1038/s41592-020-01018-x.

[19] M. Pachitariu and C. Stringer, "Cellpose 2.0: How to train your own model," Nature Methods, vol. 19, no. 12, pp. 1634--1641, 2022, doi: 10.1038/s41592-022-01663-4.

[20] M. Y. Lee et al., "CellSeg: A robust, pre-trained nucleus segmentation and pixel quantification software for highly multiplexed fluorescence images," BMC Bioinformatics, vol. 23, no. 1, p. 46, 2022, doi: 10.1186/s12859-022-04570-9.

[21] D. Schapiro et al., "MCMICRO: A scalable, modular image-processing pipeline for multiplexed tissue imaging," Nature Methods, vol. 19, no. 3, pp. 311--315, 2022, doi: 10.1038/s41592-021-01308-y.

[22] M. J. D. Baars et al., "MATISSE: A method for improved single cell segmentation in imaging mass cytometry," BMC Biology, vol. 19, no. 1, p. 99, 2021, doi: 10.1186/s12915-021-01043-y.

[23] S. Mandal and V. Uhlmann, "SplineDist: Automated Cell Segmentation With Spline Curves." Bioinformatics, 2020. doi: 10.1101/2020.10.27.357640.

[24] U. Schmidt, M. Weigert, C. Broaddus, and G. Myers, "Cell Detection with Star-convex Polygons," 2018, doi: 10.48550/ARXIV.1806.03535.

[25] V. Petukhov et al., "Cell segmentation in imaging-based spatial transcriptomics," Nature Biotechnology, vol. 40, no. 3, pp. 345--354, 2022, doi: 10.1038/s41587-021-01044-w.

[26] J. Park et al., "Cell segmentation-free inference of cell types from in situ transcriptomics data," Nature Communications, vol. 12, no. 1, p. 3545, 2021, doi: 10.1038/s41467-021-23807-4.

[27] S. Prabhakaran, "Sparcle: Assigning transcripts to cells in multiplexed images," Bioinformatics Advances, vol. 2, no. 1, p. vbac048, 2022, doi: 10.1093/bioadv/vbac048.

[28] Y. He et al., "ClusterMap for multi-scale clustering analysis of spatial gene expression," Nature Communications, vol. 12, no. 1, p. 5909, 2021, doi: 10.1038/s41467-021-26044-x.

[29] Y. Si et al., "FICTURE: Scalable segmentation-free analysis of submicron-resolution spatial transcriptomics," Nature Methods, vol. 21, no. 10, pp. 1843--1854, 2024, doi: 10.1038/s41592-024-02415-2.

[30] X. Fu et al., "BIDCell: Biologically-informed self-supervised learning for segmentation of subcellular spatial transcriptomics data," Nature Communications, vol. 15, no. 1, p. 509, 2024, doi: 10.1038/s41467-023-44560-w.

[31] T. Defard et al., "A point cloud segmentation framework for image-based spatial transcriptomics," Communications Biology, vol. 7, no. 1, p. 823, 2024, doi: 10.1038/s42003-024-06480-3.

[32] G. Partel and C. Wählby, "Spage2vec: Unsupervised representation of localized spatial gene expression signatures," The FEBS Journal, vol. 288, no. 6, pp. 1859--1870, 2021, doi: 10.1111/febs.15572.

[33] X. Qian et al., "Probabilistic cell typing enables fine mapping of closely related cell types in situ," Nature Methods, vol. 17, no. 1, pp. 101--106, 2020, doi: 10.1038/s41592-019-0631-4.

[34] S. Marco Salas et al., "Optimizing Xenium In Situ data utility by quality assessment and best-practice analysis workflows," Nature Methods, vol. 22, no. 4, pp. 813--823, 2025, doi: 10.1038/s41592-025-02617-2.

[35] V. Van Unen et al., "Visual analysis of mass cytometry data by hierarchical stochastic neighbour embedding reveals rare cell types," Nature Communications, vol. 8, no. 1, p. 1740, 2017, doi: 10.1038/s41467-017-01689-9.

[36] S. Van Gassen et al., "FlowSOM: Using self‐organizing maps for visualization and interpretation of cytometry data," CytometryPartA, vol. 87, no. 7, pp. 636--645, 2015, doi: 10.1002/cyto.a.22625.

[37] K. Quintelier, A. Couckuyt, A. Emmaneel, J. Aerts, Y. Saeys, and S. Van Gassen, "Analyzing high-dimensional cytometry data using FlowSOM," Nature Protocols, vol. 16, no. 8, pp. 3775--3801, 2021, doi: 10.1038/s41596-021-00550-0.

[38] L. Heumos et al., "Best practices for single-cell analysis across modalities," Nature Reviews Genetics, vol. 24, no. 8, pp. 550--572, 2023, doi: 10.1038/s41576-023-00586-w.

[39] V. D. Blondel, J.-L. Guillaume, R. Lambiotte, and E. Lefebvre, "Fast unfolding of communities in large networks," Journal of Statistical Mechanics: Theory and Experiment, vol. 2008, no. 10, p. P10008, 2008, doi: 10.1088/1742-5468/2008/10/P10008.

[40] L. Waltman and N. J. Van Eck, "A smart local moving algorithm for large-scale modularity-based community detection," The European Physical Journal B, vol. 86, no. 11, p. 471, 2013, doi: 10.1140/epjb/e2013-40829-0.

[41] V. A. Traag, L. Waltman, and N. J. Van Eck, "From Louvain to Leiden: Guaranteeing well-connected communities," Scientific Reports, vol. 9, no. 1, 2019, doi: 10.1038/s41598-019-41695-z.

[42] C. Domínguez Conde et al., "Cross-tissue immune cell analysis reveals tissue-specific features in humans," Science, vol. 376, no. 6594, p. eabl5197, 2022, doi: 10.1126/science.abl5197.

[43] J. B. Kang et al., "Efficient and precise single-cell reference atlas mapping with Symphony," Nature Communications, vol. 12, no. 1, p. 5890, 2021, doi: 10.1038/s41467-021-25957-x.

[44] D. Schapiro et al., "histoCAT: Analysis of cell phenotypes and interactions in multiplex image cytometry data," Nature Methods, vol. 14, no. 9, pp. 873--876, 2017, doi: 10.1038/nmeth.4391.

[45] Y. Goltsev et al., "Deep Profiling of Mouse Splenic Architecture with CODEX Multiplexed Imaging," Cell, vol. 174, no. 4, pp. 968--981.e15, 2018, doi: 10.1016/j.cell.2018.07.010.

[46] C. M. Schürch et al., "Coordinated Cellular Neighborhoods Orchestrate Antitumoral Immunity at the Colorectal Cancer Invasive Front," Cell, vol. 182, no. 5, pp. 1341--1359.e19, 2020, doi: 10.1016/j.cell.2020.07.005.

[47] D. Haviv et al., "The covariance environment defines cellular niches for spatial inference," Nature Biotechnology, vol. 43, no. 2, pp. 269--280, 2025, doi: 10.1038/s41587-024-02193-4.

[48] S. Birk et al., "Quantitative characterization of cell niches in spatially resolved omics data," Nature Genetics, vol. 57, no. 4, pp. 897--909, 2025, doi: 10.1038/s41588-025-02120-6.

[49] V. Singhal et al., "BANKSY unifies cell typing and tissue domain segmentation for scalable spatial omics data analysis," Nature Genetics, vol. 56, no. 3, pp. 431--441, 2024, doi: 10.1038/s41588-024-01664-3.

[50] E. Zhao et al., "Spatial transcriptomics at subspot resolution with BayesSpace," Nature Biotechnology, vol. 39, no. 11, pp. 1375--1384, 2021, doi: 10.1038/s41587-021-00935-2.

[51] Q. Zhu, S. Shah, R. Dries, L. Cai, and G.-C. Yuan, "Identification of spatially associated subpopulations by combining scRNAseq and sequential fluorescence in situ hybridization data," Nature Biotechnology, vol. 36, no. 12, pp. 1183--1190, 2018, doi: 10.1038/nbt.4260.

[52] D. Pham et al., "Robust mapping of spatiotemporal trajectories and cell--cell interactions in healthy and diseased tissues," Nature Communications, vol. 14, no. 1, p. 7739, 2023, doi: 10.1038/s41467-023-43120-6.

[53] J. Hu et al., "SpaGCN: Integrating gene expression, spatial location and histology to identify spatial domains and spatially variable genes by graph convolutional network," Nature Methods, vol. 18, no. 11, pp. 1342--1351, 2021, doi: 10.1038/s41592-021-01255-8.

[54] K. Dong and S. Zhang, "Deciphering spatial domains from spatially resolved transcriptomics with an adaptive graph attention auto-encoder," Nature Communications, vol. 13, no. 1, p. 1739, 2022, doi: 10.1038/s41467-022-29439-6.

[55] Y. Long et al., "Spatially informed clustering, integration, and deconvolution of spatial transcriptomics with GraphST," Nature Communications, vol. 14, no. 1, p. 1155, 2023, doi: 10.1038/s41467-023-36796-3.

[56] A. Andersson, A. Behanova, C. Avenel, J. Windhager, F. Malmberg, and C. Wählby, "Points2Regions: Fast, interactive clustering of imaging‐based spatial transcriptomics data," Cytometry Part A, vol. 105, no. 9, pp. 677--687, 2024, doi: 10.1002/cyto.a.24884.

[57] B. L. Walker and Q. Nie, "NeST: Nested hierarchical structure identification in spatial transcriptomic data," Nature Communications, vol. 14, no. 1, 2023, doi: 10.1038/s41467-023-42343-x.

[58] S. S. Bhate, G. L. Barlow, C. M. Schürch, and G. P. Nolan, "Tissue schematics map the specialization of immune tissue motifs and their appropriation by tumors," Cell Systems, vol. 13, no. 2, pp. 109--130.e6, 2022, doi: 10.1016/j.cels.2021.09.012.

[59] A. Getis, "Spatial Autocorrelation," in Handbook of Applied Spatial Analysis, M. M. Fischer and A. Getis, Eds., Berlin, Heidelberg: Springer Berlin Heidelberg, 2010, pp. 255--278. doi: 10.1007/978-3-642-03647-7_14.

[60] V. Svensson, S. A. Teichmann, and O. Stegle, "SpatialDE: Identification of spatially variable genes," Nature Methods, vol. 15, no. 5, pp. 343--346, 2018, doi: 10.1038/nmeth.4636.

[61] D. Edsgärd, P. Johnsson, and R. Sandberg, "Identification of spatial expression trends in single-cell gene expression data," Nature Methods, vol. 15, no. 5, pp. 339--342, 2018, doi: 10.1038/nmeth.4634.

[62] S. Sun, J. Zhu, and X. Zhou, "Statistical analysis of spatial expression patterns for spatially resolved transcriptomic studies," Nature Methods, vol. 17, no. 2, pp. 193--200, 2020, doi: 10.1038/s41592-019-0701-7.

[63] J. Zhu, S. Sun, and X. Zhou, "SPARK-X: Non-parametric modeling enables scalable and robust detection of spatial expression patterns for large spatial transcriptomic studies," Genome Biology, vol. 22, no. 1, p. 184, 2021, doi: 10.1186/s13059-021-02404-0.

[64] A. Andersson and J. Lundeberg, "Sepal: Identifying transcript profiles with spatial patterns by diffusion-based modeling," Bioinformatics, vol. 37, no. 17, pp. 2644--2650, 2021, doi: 10.1093/bioinformatics/btab164.

[65] K. Zhang, W. Feng, and P. Wang, "Identification of spatially variable genes with graph cuts," Nature Communications, vol. 13, no. 1, p. 5488, 2022, doi: 10.1038/s41467-022-33182-3.

[66] E. Armingol, H. M. Baghdassarian, and N. E. Lewis, "The diversification of methods for studying cell--cell interactions and communication," Nature Reviews Genetics, vol. 25, no. 6, pp. 381--400, 2024, doi: 10.1038/s41576-023-00685-8.

[67] D. Dimitrov et al., "LIANA+ provides an all-in-one framework for cell--cell communication inference," Nature Cell Biology, vol. 26, no. 9, 2024, doi: 10.1038/s41556-024-01469-w.

[68] D. S. Fischer, A. C. Schaar, and F. J. Theis, "Modeling intercellular communication in tissues using spatial graphs of cells," Nature Biotechnology, vol. 41, no. 3, pp. 332--336, 2023, doi: 10.1038/s41587-022-01467-z.

[69] K. Troulé et al., "CellPhoneDB v5: Inferring cell--cell communication from single-cell multiomics data," Nature Protocols, 2025, doi: 10.1038/s41596-024-01137-1.

[70] S. Jin, M. V. Plikus, and Q. Nie, "CellChat for systematic analysis of cell--cell communication from single-cell transcriptomics," Nature Protocols, vol. 20, no. 1, pp. 180--219, 2025, doi: 10.1038/s41596-024-01045-4.

[71] X. Shao et al., "Knowledge-graph-based cell-cell communication inference for spatially resolved transcriptomic data with SpaTalk," Nature Communications, vol. 13, no. 1, p. 4429, 2022, doi: 10.1038/s41467-022-32111-8.

[72] Z. Cang et al., "Screening cell--cell communication in spatial transcriptomics via collective optimal transport," Nature Methods, vol. 20, no. 2, 2023, doi: 10.1038/s41592-022-01728-4.

[73] Z. Li, T. Wang, P. Liu, and Y. Huang, "SpatialDM for rapid identification of spatially co-expressed ligand--receptor and revealing cell--cell communication patterns," Nature Communications, vol. 14, no. 1, p. 3995, 2023, doi: 10.1038/s41467-023-39608-w.

[74] R. Browaeys, W. Saelens, and Y. Saeys, "NicheNet: Modeling intercellular communication by linking ligands to target genes," Nature Methods, vol. 17, no. 2, pp. 159--162, 2020, doi: 10.1038/s41592-019-0667-5.

[75] Y. Yuan and Z. Bar-Joseph, "GCNG: Graph convolutional networks for inferring gene interaction from spatial transcriptomics data," Genome Biology, vol. 21, no. 1, 2020, doi: 10.1186/s13059-020-02214-w.

[76] J. Zhu et al., "Mapping cellular interactions from spatially resolved transcriptomics data," Nature Methods, vol. 21, no. 10, pp. 1830--1842, 2024, doi: 10.1038/s41592-024-02408-1.

[77] B. L. Walker, Z. Cang, H. Ren, E. Bourgain-Chang, and Q. Nie, "Deciphering tissue structure and function using spatial transcriptomics," Communications Biology, vol. 5, no. 1, p. 220, 2022, doi: 10.1038/s42003-022-03175-5.

[78] D. Türei, T. Korcsmáros, and J. Saez-Rodriguez, "OmniPath: Guidelines and gateway for literature-curated signaling pathway resources," Nature Methods, vol. 13, no. 12, pp. 966--967, 2016, doi: 10.1038/nmeth.4077.

[79] D. Arnol, D. Schapiro, B. Bodenmiller, J. Saez-Rodriguez, and O. Stegle, "Modeling Cell-Cell Interactions from Spatial Molecular Data with Spatial Variance Component Analysis," Cell Reports, vol. 29, no. 1, 2019, doi: 10.1016/j.celrep.2019.08.077.

[80] J. Tanevski, R. O. R. Flores, A. Gabor, D. Schapiro, and J. Saez-Rodriguez, "Explainable multiview framework for dissecting spatial relationships from highly multiplexed data," Genome Biology, vol. 23, no. 1, 2022, doi: 10.1186/s13059-022-02663-5.

[81] P. Badia-i-Mompel et al., "decoupleR: Ensemble of computational methods to infer biological activities from omics data," Bioinformatics Advances, vol. 2, no. 1, p. vbac016, 2022, doi: 10.1093/bioadv/vbac016.

[82] M. Schubert et al., "Perturbation-response genes reveal signaling footprints in cancer gene expression," Nature Communications, vol. 9, no. 1, p. 20, 2018, doi: 10.1038/s41467-017-02391-6.

[83] S. Müller-Dott et al., "Expanding the coverage of regulons from high-confidence prior knowledge for accurate estimation of transcription factor activities," Nucleic Acids Research, vol. 51, no. 20, pp. 10934--10949, 2023, doi: 10.1093/nar/gkad841.

[84] G. Palla et al., "Squidpy: A scalable framework for spatial omics analysis," Nature Methods, vol. 19, no. 2, pp. 171--178, 2022, doi: 10.1038/s41592-021-01358-2.

[85] R. Argelaguet et al., "Multi‐Omics Factor Analysis---a framework for unsupervised integration of multi‐omics data sets," Molecular Systems Biology, vol. 14, no. 6, p. e8124, 2018, doi: 10.15252/msb.20178124.

[86] R. Argelaguet et al., "MOFA+: A statistical framework for comprehensive integration of multi-modal single-cell data," Genome Biology, vol. 21, no. 1, 2020, doi: 10.1186/s13059-020-02015-1.

[87] R. O. Ramirez Flores, J. D. Lanzer, D. Dimitrov, B. Velten, and J. Saez-Rodriguez, "Multicellular factor analysis of single-cell data for a tissue-centric understanding of disease," eLife, vol. 12, p. e93161, 2023, doi: 10.7554/eLife.93161.

[88] B. Velten et al., "Identifying temporal and spatial patterns of variation from multimodal data using MEFISTO," Nature Methods, vol. 19, no. 2, pp. 179--186, 2022, doi: 10.1038/s41592-021-01343-9.

[89] L. Jerby-Arnon and A. Regev, "DIALOGUE maps multicellular programs in tissue from single-cell or spatial transcriptomics data," Nature Biotechnology, vol. 40, no. 10, pp. 1467--1477, 2022, doi: 10.1038/s41587-022-01288-0.

[90] J. Mitchel et al., "Coordinated, multicellular patterns of transcriptional variation that stratify patient cohorts are revealed by tensor decomposition," Nature Biotechnology, 2024, doi: 10.1038/s41587-024-02411-z.

[91] F. W. Townes and B. E. Engelhardt, "Nonnegative spatial factorization applied to spatial genomics," Nature Methods, vol. 20, no. 2, pp. 229--238, 2023, doi: 10.1038/s41592-022-01687-w.

[92] D. Jiménez-Sánchez, M. Ariz, H. Chang, X. Matias-Guiu, C. E. De Andrea, and C. Ortiz-de-Solórzano, "NaroNet: Discovery of tumor microenvironment elements from highly multiplexed images," Medical Image Analysis, vol. 78, p. 102384, 2022, doi: 10.1016/j.media.2022.102384.

[93] X. Fu, E. Patrick, J. Y. H. Yang, D. D. Feng, and J. Kim, "Deep multimodal graph-based network for survival prediction from highly multiplexed images and patient variables," Computers in Biology and Medicine, vol. 154, p. 106576, 2023, doi: 10.1016/j.compbiomed.2023.106576.

[94] T. Risom et al., "Transition to invasive breast cancer is associated with progressive changes in the structure and composition of tumor stroma," Cell, vol. 185, no. 2, pp. 299--310.e18, 2022, doi: 10.1016/j.cell.2021.12.023.

[95] S. Babaei et al., "S3-CIMA: Supervised spatial single-cell image analysis for identifying disease-associated cell-type compositions in tissue," Patterns, vol. 4, no. 9, p. 100829, 2023, doi: 10.1016/j.patter.2023.100829.

[96] Y. Wang et al., "Cell graph neural networks enable the precise prediction of patient survival in gastric cancer," npj Precision Oncology, vol. 6, no. 1, p. 45, 2022, doi: 10.1038/s41698-022-00285-5.

[97] R. Perez-Lopez, N. Ghaffari Laleh, F. Mahmood, and J. N. Kather, "A guide to artificial intelligence for cancer researchers," Nature Reviews Cancer, vol. 24, no. 6, pp. 427--441, 2024, doi: 10.1038/s41568-024-00694-7.

[98] M. Unger and J. N. Kather, "A systematic analysis of deep learning in genomics and histopathology for precision oncology," BMC Medical Genomics, vol. 17, no. 1, p. 48, 2024, doi: 10.1186/s12920-024-01796-9.

[99] Y. Hao et al., "Integrated analysis of multimodal single-cell data," Cell, vol. 184, no. 13, pp. 3573--3587.e29, 2021, doi: 10.1016/j.cell.2021.04.048.

[100] F. A. Wolf, P. Angerer, and F. J. Theis, "SCANPY: Large-scale single-cell gene expression data analysis," Genome Biology, vol. 19, no. 1, p. 15, 2018, doi: 10.1186/s13059-017-1382-0.

[101] I. Virshup, S. Rybakov, F. J. Theis, P. Angerer, and F. A. Wolf, "Anndata: Access and store annotated datamatrices," Journal of Open Source Software, vol. 9, no. 101, p. 4371, 2024, doi: 10.21105/joss.04371.

[102] R. Dries et al., "Giotto: A toolbox for integrative analysis and visualization of spatial expression data," Genome Biology, vol. 22, no. 1, p. 78, 2021, doi: 10.1186/s13059-021-02286-2.

[103] L. Marconato et al., "SpatialData: An open and universal data framework for spatial omics," Nature Methods, vol. 22, no. 1, pp. 58--62, 2025, doi: 10.1038/s41592-024-02212-x.

[104] N. Sofroniew et al., "Napari: A multi-dimensional image viewer for Python." Zenodo, 2025. doi: 10.5281/ZENODO.3555620.

[105] L. Moses and L. Pachter, "Museum of spatial transcriptomics," Nature Methods, vol. 19, no. 5, pp. 534--546, 2022, doi: 10.1038/s41592-022-01409-2.

[106] E. Roesch et al., "Julia for biologists," Nature Methods, vol. 20, no. 5, pp. 655--664, 2023, doi: 10.1038/s41592-023-01832-z.

[107] I. Virshup et al., "The scverse project provides a computational ecosystem for single-cell omics data analysis," Nature Biotechnology, vol. 41, no. 5, 2023, doi: 10.1038/s41587-023-01733-8.

[108] C. C. Liu et al., "Robust phenotyping of highly multiplexed tissue imaging data using pixel-level clustering," Nature Communications, vol. 14, no. 1, 2023, doi: 10.1038/s41467-023-40068-5.