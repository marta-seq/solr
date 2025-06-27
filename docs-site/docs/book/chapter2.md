---
bibliography: bibliography.bib
csl: apa.csl
---

# Analytical Frameworks for Spatial Omics Data

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
(Baharlou et al., 2019; Baranski et al., 2021; Chevrier et al., 2018)
and may also exhibit low signal-to-noise ratios (Milosevic, 2023).

Channel crosstalk, where signals from one channel interfere with
adjacent channels, can be mitigated through well-designed antibody
panels (Milosevic, 2023) or correction methods like CATALYST (Chevrier
et al., 2018), which uses pre-acquisition compensation matrices, as well
as post-processing techniques (Baranski et al., 2021; Y. J. Wang et al.,
2019).

Beyond crosstalk, other sources of noise---such as hot pixels,
background signal, and shot noise---require distinct computational
strategies. Some studies have employed adjacent approaches, including
those implemented in Steinbock (Windhager et al., 2023) and MAUI (MBI
analysis interface) (Baranski et al., 2021), while others have developed
"homebrew" methods based on traditional image filtering techniques
(Keren et al., 2018; Rendeiro et al., 2021; Y. J. Wang et al., 2019).
However, these methods are often insufficient for fully addressing
complex noise patterns.

Tools like Ilastik (Berg et al., 2019) provide supervised pixel
classification to distinguish background noise from true signal on a
per-marker basis (Berg et al., 2019; Ijsselsteijn et al., 2021;
Windhager et al., 2023). Although this approach requires extensive
manual annotation, it is currently considered state-of-the-art, as it
effectively removes background noise and improves signal normalization
and batch effect reduction across samples (Keren et al., 2018). More
recently, IMC-Denoise (Lu et al., 2023) has been introduced as a
two-step pipeline combining traditional algorithms with a
self-supervised deep learning model based on Noise2Void (Krull et al.,
2019).

Notably, this thesis contributes to this specific challenge, as
discussed in Chapter [\[ch:penguin\]](#ch:penguin){reference-type="ref"
reference="ch:penguin"}.

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
molecular measurements---transcripts or proteins---to individual cells.
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
methods are implemented in established tools like CellProfiler
(Carpenter et al., 2006; McQuin et al., 2018) and ImageJ/Fiji
(Schindelin et al., 2012), which allow customization of segmentation
pipelines using scripting interfaces. In recent years, DL models have
gained prominence due to their improved accuracy, generalizability, and
community availability (Von Chamier et al., 2021). Notably, Mesmer, part
of the DeepCell framework, DL-based model specifically trained on
spatial proteomics datasets, leveraging TissueNet, a large annotated
dataset (Greenwald et al., 2022). Another widely adopted method is
Cellpose, a generalist DL segmentation tool that predicts vector flows
to delineate cell boundaries instead of relying on direct pixel-wise
classification. Though originally trained on a broad array of microscopy
images, Cellpose also offers specialist models, including those
fine-tuned on TissueNet subsets to enhance performance on spatial
proteomics images (Pachitariu & Stringer, 2022; Stringer et al., 2021).

Segmentation strategies in spatial proteomics range from nucleus-based
expansion methods, which approximate cell outlines from DAPI staining,
to full-cell segmentation techniques that incorporate both nuclear and
membrane markers. A growing number of domain-specific models continue to
emerge, each tailored to address the distinct noise profiles, spatial
resolution, and multiplexing characteristics of spatial proteomics data
(Baars et al., 2021; Lee et al., 2022; Mandal & Uhlmann, 2020; Schapiro
et al., 2022; Schmidt et al., 2018).

This thesis also contributes to this domain, as discussed in
Chapter [\[ch:Cellcytolego\]](#ch:Cellcytolego){reference-type="ref"
reference="ch:Cellcytolego"}.

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
dependency on auxiliary images (Defard et al., 2024; Fu et al., 2024; He
et al., 2021; Park et al., 2021; Partel & Wählby, 2021; Petukhov et al.,
2022; Prabhakaran, 2022; Qian et al., 2020; Si et al., 2024).

A prominent example is Baysor (Petukhov et al., 2022), a general
probabilistic framework implemented in Julia. Based on
[MRF]{acronym-label="MRF" acronym-form="singular+short"}, Baysor models
cells by jointly considering spatial proximity and transcriptional
similarity. It can incorporate prior segmentations, assigning confidence
scores to them, or operate de novo using only transcript data. Baysor
functions in both 2D and 3D, optimizing the likelihood that neighbouring
transcripts originate from the same cell. Notably, it has shown strong
performance across datasets and has been reported to outperform the
default segmentation provided in Xenium workflows (Marco Salas et al.,
2025), highlighting its practical utility and growing adoption.

## Cell phenotyping

Cell phenotyping is the process of assigning biological identities to
individual cells based on their molecular profiles, such as gene or
protein expression, and serves as a critical step in interpreting
spatial omics data.

### Spatial proteomics

Unlike transcriptomic approaches, spatial proteomics typically profiles
a pre-selected panel of $\sim$`<!-- -->`{=html}30--50 protein markers,
chosen based on the biological question. This reduced and curated
feature space simplifies downstream computational analysis but increases
reliance on prior biological knowledge for interpreting clusters
(Windhager et al., 2023).

Commonly used clustering algorithms include Cytosplore, a GUI which
leverages [HSNE]{acronym-label="HSNE" acronym-form="singular+short"}(Van
Unen et al., 2017), Phenograph that builds a shared nearest neighbor
graph and uses Louvain algorithm to partition of the graph into
communities (Van Unen et al., 2017), and FlowSOM that applies
[SOM]{acronym-label="SOM" acronym-form="singular+short"} (Quintelier et
al., 2021; Van Gassen et al., 2015).

In addition to unsupervised clustering, other strategies for phenotyping
include: Manual gating, where marker thresholds are used to define known
populations based on expert knowledge; Supervised machine learning
models, such as Random Forests, which classify cells using labelled
training data and reference mapping, where new datasets are aligned to
previously annotated references to infer phenotypes.

### Spatial transcriptomics

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
structure (Heumos et al., 2023).

Next, clustering algorithms group similar cells together. This is
usually done by building a [KNN]{acronym-label="KNN"
acronym-form="singular+short"} graph in the PCA-reduced space. Each cell
is connected to its K most similar neighbours, and clustering algorithms
such as Louvain (Blondel et al., 2008) and Leiden (Waltman & Van Eck,
2013), often with better performance (Traag et al., 2019) are applied to
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
specificity of known markers (Heumos et al., 2023).

Alternatively, automated annotation methods are also available, which
leverage reference gene atlases and machine/deep learning models (e.g.,
CellTypist (Domínguez Conde et al., 2022)), or label transfer approaches
(e.g., Symphony (Kang et al., 2021)). These methods provide faster and
more scalable annotation but tend to be less interpretable and dependent
on the similarity between the reference and the query data (Heumos et
al., 2023).

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
differences in the underlying molecular readouts (Heumos et al., 2023;
Windhager et al., 2023).

Once a graph is established, it enables the exploration of spatial
interactions between annotated cell types. A common approach involves
the computation of neighbourhood enrichment scores, which statistically
test whether specific cell types are found adjacent to each other more
or less frequently than expected by chance. These analyses rely on
permutation-based null models, which shuffle cell-type labels while
preserving tissue structure, providing a robust statistical framework to
assess enrichment or depletion of interactions (Schapiro et al., 2017).
Another related method is the computation of co-occurrence scores, which
estimate how likely it is to observe specific cell-type pairs within
increasing radii around each cell. These scores reflect the conditional
probability of observing a certain cell type given the presence of
another nearby, offering an interpretable measure of spatial association
across scales (Heumos et al., 2023).

Alternatively, simple interaction matrices can be computed, summarizing
the raw counts of neighbouring cell-type pairs (Windhager et al., 2023).
Though not statistical tests, these matrices are useful for exploratory
data analysis.

Beyond pairwise interaction analyses, more integrative approaches aim to
cluster cells based on the composition of their local neighbourhoods.
One strategy involves computing, for each cell, the fraction of
surrounding cell types within its local environment (e.g., 20 nearest
neighbours). These local composition profiles can then be clustered
using unsupervised methods such as k-means or Leiden algorithms,
grouping cells into recurring spatial neighbourhoods. This approach was
introduced by studies such as Goltsev et al. (Goltsev et al., 2018) and
Schürch et al. (Schürch et al., 2020), which revealed that specific
combinations of neighbouring discrete cell types are often spatially
organized in conserved patterns.

A related method involves aggregating gene or protein expression
features across the neighbourhood, effectively capturing a summary of
the local microenvironment. These aggregated features can then serve as
the basis for clustering or downstream modelling, this is approach is
implemented for example in covariance environment (COVET) (Haviv et al.,
2025), NicheCompass, a graph deep-learning approach to identify and
quantitatively characterize niches by learning cell embeddings encoding
signalling events as spatial gene program activities (Birk et al., 2025)
and Banksy (Singhal et al., 2024).

Through these computational strategies, spatial neighbourhoods can be
identified and interpreted biologically as spatial niches---localized,
recurring configurations of cells and their molecular environment that
reflect functionally relevant tissue microenvironments. Spatial niches
are particularly informative in contexts such as immuno-oncology, where
immune-tumour interactions shape the progression or suppression of
disease.

## Spatial domains

Closely related to cellular neighbourhoods, spatial domains usually
refer to broader, tissue-scale regions characterized by coherent gene or
protein expression patterns and underlying structural organization
(Heumos et al., 2023). While cellular neighbourhoods capture the
immediate microenvironment around a cell---defined by local interactions
and spatial proximity---spatial domains extend this concept to larger
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
to be assigned the same label (Zhao et al., 2021; Q. Zhu et al., 2018).
This assumes that a cell's transcriptome resembles the average of its
domain, although this may not always hold true in tissues where multiple
cell types are intermixed. Other approaches employ deep learning,
particularly graph-based neural networks that can incorporate either
histological information (Hu et al., 2021; Pham et al., 2023) or spatial
graphs derived purely from molecular and positional data (Andersson et
al., 2024; Dong & Zhang, 2022; Long et al., 2023). These methods offer
flexibility and can capture complex spatial dependencies, but their
performance can vary with dataset size and structure.

More recently, methods introducing hierarchical and multiscale
representations of spatial structure have emerged. For instance, NeST
(Walker & Nie, 2023) identifies nested co-expression
hotspots---spatially contiguous regions that co-express subsets of
genes---by simultaneously searching across gene and spatial dimensions.
This approach captures biological structure at multiple scales and can
reveal overlapping or nested domains, without requiring prior
assumptions about gene sets or spatial resolution. Similarly, the
concept of tissue schematics (Bhate et al., 2022) has been proposed as a
way to abstract spatial organization into higher-order motifs, by
identifying modular assemblies of cellular neighbourhoods. These
schematics can be used to represent tissue architecture in both healthy
and diseased states, offering insights into how local interactions scale
up to tissue-level functionality.

## Spatially Variable genes

[SVG]{acronym-label="SVG" acronym-form="singular+short"}s are genes
whose expression displays significant spatial patterning across a
tissue. Identifying SVGs reveals insights into tissue architecture,
cell--cell communication, and local microenvironmental influences,
extending beyond what is captured by cell types or spatial domains
alone. These spatial patterns may arise from gradients in signalling
molecules, regional functional differences, or heterogeneous cell
compositions (Heumos et al., 2023). Notably, SVG detection does not rely
on cell segmentation, making it applicable across different spatial
omics technologies.

Several computational methods have been developed to detect SVGs by
decomposing gene expression variability into spatial and non-spatial
components. A widely used statistic is Moran's I, which quantifies
spatial autocorrelation by measuring how similar gene expression is
between neighbouring spots (Getis, 2010). Model-based approaches like
SpatialDE (Svensson et al., 2018), trendseek (Edsgärd et al., 2018),
SPARK (Sun et al., 2020), and SPARK-X (Zhu et al., 2021) use Gaussian
processes or spatial statistical models to test for spatially structured
expression while accounting for noise and overdispersion. While these
tools test each gene independently and return p-values, they often
overlook spatial domain context, which can limit biological
interpretability. Other strategies take different modelling
perspectives: Sepal (Andersson & Lundeberg, 2021) applies a Gaussian
diffusion process, scGCO (Zhang et al., 2022) uses graph cuts to detect
spatial expression boundaries, and SpaGCN (Hu et al., 2021), defines
both domains and SVGs integrating histology and spatial proximity
through graph convolutional networks.

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
investigating intracellular signalling pathways, [TF]{acronym-label="TF"
acronym-form="singular+short"} activity, and particularly
c[CCC]{acronym-label="CCC" acronym-form="singular+short"} events---often
inferred from curated interaction networks derived from transcriptomics
data (Armingol et al., 2024; Dimitrov et al., 2024).

A broad landscape of computational tools has emerged for CCC analysis
(Browaeys et al., 2020; Cang et al., 2023; Dimitrov et al., 2024;
Fischer et al., 2023; Jin et al., 2025; Li et al., 2023; Shao et al.,
2022; Troulé et al., 2025; Yuan & Bar-Joseph, 2020; Zhu et al., 2024),
reflecting the growing complexity and richness of spatial omics data.
Broadly, these methods fall into two sections: identifying pairs of
genes that interact, such that expression of the gene in one cell
influences that of the other gene in others; and identifying pairs of
cells in which that gene pair interacts (Walker et al., 2022).

Most methods for CCC leverage [L-R]{acronym-label="L-R"
acronym-form="singular+short"} interactions, relying heavily on prior
biological knowledge, often curated into databases such as CellPhoneDB
(Troulé et al., 2025), OmniPath (Türei et al., 2016), and CellChat (Jin
et al., 2025). These databases catalogue known signalling pathways and
interaction pairs from the literature. Using such resources, CCC
inference can be extended to the spatial context by identifying L--R
pairs co-expressed in nearby cells (Dimitrov et al., 2024) for example
using spatial co-expression (Dimitrov et al., 2024; Li et al., 2023), or
incorporating different approaches such as optimal transport frameworks
(Cang et al., 2023) or Bayesian [MIL]{acronym-label="MIL"
acronym-form="singular+short"}(Zhu et al., 2024) to more precisely link
cells by both location and expression.

Complementing these prior knowledge-driven approaches are data-driven
methods that model spatial gene expression to discover novel
interactions. These models, like NCEM(Fischer et al., 2023), GCNG(Yuan &
Bar-Joseph, 2020) or SVCA(Arnol et al., 2019) and Misty(Tanevski et al.,
2022) (while not specific for CCC), aim to capture interactions that
explain spatial expression variance across multiple genes. They enable
inference of new communication patterns not captured in curated L--R
databases, though their accuracy depends on model design and training
data.

Furthermore, some CCC frameworks estimate global co-localization across
entire tissue sections (Fischer et al., 2023; Shao et al., 2022;
Tanevski et al., 2022), and others focus on local cell-cell proximity
(Cang et al., 2023; Li et al., 2023).

In parallel, there is increasing interest in inferring intracellular
functional activity---such as pathway activation, TF activity, or gene
set enrichment---within individual cells or regions. Tools like
decouplR(Badia-i-Mompel et al., 2022) apply prior knowledge from pathway
perturbation signatures(Schubert et al., 2018) or regulatory
networks(Müller-Dott et al., 2023) to estimate these functional states
at single-cell or spot level. As with CCC, these functional scores can
then be spatially mapped. These approaches are currently more mature for
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
differences between conditions (Dimitrov et al., 2024; Palla et al.,
2022; Tanevski et al., 2022). However, this strategy is often too
simplistic, as it can blur important heterogeneity by averaging out
spatially localized effects.

### Matrix factorization methods

Matrix factorization methods provide a more nuanced alternative. These
approaches reduce high-dimensional data into a smaller set of latent
factors that capture major patterns of variation. Classical approaches
like PCA achieve this through linear decompositions, but newer tools
like [MOFA]{acronym-label="MOFA"
acronym-form="singular+short"}(Argelaguet et al., 2018, 2020; Ramirez
Flores et al., 2023), MEFISTO(Velten et al., 2022), and
DIALOGUE(Jerby-Arnon & Regev, 2022) extend the concept to handle
multiple data types, structured variation, or intercellular
coordination. These methods can incorporate spatial information embedded
in the matrix.

MOFA and its variants allow joint analysis of multi-omic datasets,
identifying latent factors that capture both shared and group-specific
variation. Other tools like DIALOGUE(Jerby-Arnon & Regev, 2022) and
scITD(Mitchel et al., 2024) focus on identifying coordinated gene
programs across multiple cell types, enabling the characterization of
multicellular processes in an unsupervised fashion.

[NMF]{acronym-label="NMF" acronym-form="singular+short"} provides a more
interpretable, parts-based representation, especially useful for
identifying additive biological signals like distinct cell states or
spatial domains. Spatially-aware versions of NMF, such as
[NSF]{acronym-label="NSF" acronym-form="singular+short"}, extend this
idea by directly incorporating spatial coordinates, better capturing
localized gene expression patterns (Townes & Engelhardt, 2023).

These matrix-based methods not only enhance the ability to uncover
biologically relevant spatial and intercellular patterns, but also
enable unsupervised identification of group-level differences---whether
driven by disease, environment, or treatment---without requiring
explicit labels.

### Machine Learning and Deep Learning approaches

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

Among early efforts in this direction, NaroNet (Jiménez-Sánchez et al.,
2022) introduced a patch contrastive learning strategy combined with
graph representations to predict patient outcomes from IMC data. In the
same cohort, Fu et al.(Fu et al., 2023) proposed a deep multimodal
graph-based network that integrates IMC data with clinical variables to
predict cancer survival. In another example, Risom et al.(Risom et al.,
2022) applied a random forest classifier to a MIBI dataset of ductal
carcinoma in situ (DCIS), using over 400 spatial features---including
tissue compartment enrichment and TME morphometrics---to distinguish
between progressors and non-progressors.

A more recent and notable advance is S3-CIMA (Babaei et al., 2023), a
weakly supervised, single-layer convolutional neural network that learns
disease-specific spatial compositions of the tumour microenvironment. By
modelling local cell-type organizations from high-dimensional proteomic
imaging data (IMC and CODEX), it enables the discovery of
outcome-associated microenvironments in colorectal cancer. Similarly,
graph deep learning has been used to predict prognosis in gastric
cancer, where Cell-Graphs built from multiplexed immunohistochemistry
(mIHC) data enable prognosis prediction from spatial arrangements of
cell types (Y. Wang et al., 2022).

Important to note is that DL methods are already well-established in
computational pathology and digital histopathology, where large
annotated datasets and well-defined visual features have allowed CNNs to
thrive in image classification, segmentation, and prognosis prediction
tasks (Perez-Lopez et al., 2024; Unger & Kather, 2024).

## Frameworks and tools for spatial omics analysis

A growing ecosystem of software tools supports the analysis of
single-cell and spatial omics data. Seurat (R) (Hao et al., 2021) and
Scanpy (Python) (Wolf et al., 2018) are widely used for single-cell
analysis. Scanpy is built around the efficient AnnData structure
(Virshup et al., 2024), while Seurat uses its own SeuratObject.
Dedicated spatial omics tools such as Giotto (R) (Dries et al., 2021)
and Squidpy (Python) (Palla et al., 2022) offer integrated workflows for
spatial statistics, neighbourhood analysis, and visualization with also
new data frameworks like SpatialData (Python) (Marconato et al., 2025)
emerging to better support spatial modalities. Napari (Sofroniew et al.,
2025), a general-purpose image viewer, complements these tools with
interactive, high-dimensional image visualization---useful for working
with spatial coordinates and tissue images.

Python and R remain the dominant programming languages in this space.
Python is increasingly favoured for spatial omics due to its speed,
memory efficiency, and compatibility with deep learning and image
processing (e.g., PyTorch), while R remains popular for exploratory
analysis and visualization(Moses & Pachter, 2022). Meanwhile, Julia is
gaining attention for its performance, with tools like Baysor (Petukhov
et al., 2022) highlighting its potential in spatial omics analysis
(Roesch et al., 2023).

Given the rapid growth of tools---and the risk of incompatibilities in
data formats, APIs, and user interfaces---standardization efforts like
scverse (Virshup et al., 2023) have emerged. These initiatives promote
well-maintained, interoperable core functionality, supporting a more
cohesive and collaborative spatial omics software ecosystem.

## Closing remarks

This review has focused on spatial omics methods that operate on the
basis of cell-defined units. However, it is important to note that many
of these approaches can also be applied at the transcript level,
bypassing the need for explicit cell segmentation---this is particularly
true for tasks such as domain identification. While cell-based analysis
remains the prevailing standard in spatial omics, alternative strategies
that complement or replace cell segmentation are gaining ground
(Andersson et al., 2024; Liu et al., 2023). This shift is partly driven
by the technical difficulty of accurately segmenting individual cells,
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
Figure [1](#fig:review){reference-type="ref" reference="fig:review"}
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
need for \"living reviews\"(Moses & Pachter, 2022), curated
repositories, and community-driven benchmarking (Unger & Kather, 2024)
efforts that can adapt to the field's rapid progress.

<figure id="fig:review">
<embed src="Chapters/background/review.pdf" />
<figcaption>Graph overview of spatial omics methods analysed in this
review. Each node represents a method included in this review, with the
colour indicating its role in the spatial omics analysis pipeline (e.g.,
orange – Preprocessing, blue – Cell Segmentation, etc.) and the size of
the bubble proportional to its citation count, reflecting community
adoption. Nodes are grouped spatially by conceptual similarity and
method type. Only full spatial omics methods are shown, excluding
foundational algorithms or reused modules from unrelated pipelines. This
layout reveals the landscape and complexity of spatial omics
development. The interactive version of this graph is available at: <a
href="https://marta-seq.github.io/SOME/"
class="uri">https://marta-seq.github.io/SOME/</a>. This approach can be
extended to incorporate newly published methods or customized for other
use cases.</figcaption>
</figure>

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: {#refs .references .csl-bib-body .hanging-indent entry-spacing="0" line-spacing="2"}
::: {#ref-G14_P2R .csl-entry}
Andersson, A., Behanova, A., Avenel, C., Windhager, J., Malmberg, F., &
Wählby, C. (2024). Points2Regions: Fast, interactive clustering of
imaging‐based spatial transcriptomics data. *Cytometry Part A*,
*105*(9), 677--687. <https://doi.org/10.1002/cyto.a.24884>
:::

::: {#ref-B136_sepal_2021 .csl-entry}
Andersson, A., & Lundeberg, J. (2021). *Sepal* : Identifying transcript
profiles with spatial patterns by diffusion-based modeling.
*Bioinformatics*, *37*(17), 2644--2650.
<https://doi.org/10.1093/bioinformatics/btab164>
:::

::: {#ref-B158_mofa+2020 .csl-entry}
Argelaguet, R., Arnol, D., Bredikhin, D., Deloro, Y., Velten, B.,
Marioni, J. C., & Stegle, O. (2020). MOFA+: A statistical framework for
comprehensive integration of multi-modal single-cell data. *Genome
Biology*, *21*(1). <https://doi.org/10.1186/s13059-020-02015-1>
:::

::: {#ref-B157_MOFA2018 .csl-entry}
Argelaguet, R., Velten, B., Arnol, D., Dietrich, S., Zenz, T., Marioni,
J. C., Buettner, F., Huber, W., & Stegle, O. (2018). Multi‐Omics Factor
Analysis---a framework for unsupervised integration of multi‐omics data
sets. *Molecular Systems Biology*, *14*(6), e8124.
<https://doi.org/10.15252/msb.20178124>
:::

::: {#ref-B138_armingol2024 .csl-entry}
Armingol, E., Baghdassarian, H. M., & Lewis, N. E. (2024). The
diversification of methods for studying cell--cell interactions and
communication. *Nature Reviews Genetics*, *25*(6), 381--400.
<https://doi.org/10.1038/s41576-023-00685-8>
:::

::: {#ref-B151_svca2019 .csl-entry}
Arnol, D., Schapiro, D., Bodenmiller, B., Saez-Rodriguez, J., & Stegle,
O. (2019). Modeling Cell-Cell Interactions from Spatial Molecular Data
with Spatial Variance Component Analysis. *Cell Reports*, *29*(1).
<https://doi.org/10.1016/j.celrep.2019.08.077>
:::

::: {#ref-B95_matisse2021 .csl-entry}
Baars, M. J. D., Sinha, N., Amini, M., Pieterman-Bos, A., Van Dam, S.,
Ganpat, M. M. P., Laclé, M. M., Oldenburg, B., & Vercoulen, Y. (2021).
MATISSE: A method for improved single cell segmentation in imaging mass
cytometry. *BMC Biology*, *19*(1), 99.
<https://doi.org/10.1186/s12915-021-01043-y>
:::

::: {#ref-B167_s3cima2023 .csl-entry}
Babaei, S., Christ, J., Sehra, V., Makky, A., Zidane, M.,
Wistuba-Hamprecht, K., Schürch, C. M., & Claassen, M. (2023). S3-CIMA:
Supervised spatial single-cell image analysis for identifying
disease-associated cell-type compositions in tissue. *Patterns*, *4*(9),
100829. <https://doi.org/10.1016/j.patter.2023.100829>
:::

::: {#ref-B153_decoupler2022 .csl-entry}
Badia-i-Mompel, P., Vélez Santiago, J., Braunger, J., Geiss, C.,
Dimitrov, D., Müller-Dott, S., Taus, P., Dugourd, A., Holland, C. H.,
Ramirez Flores, R. O., & Saez-Rodriguez, J. (2022).
[decoupleR]{.nocase}: Ensemble of computational methods to infer
biological activities from omics data. *Bioinformatics Advances*,
*2*(1), vbac016. <https://doi.org/10.1093/bioadv/vbac016>
:::

::: {#ref-P6_baharlou_mass_2019 .csl-entry}
Baharlou, H., Canete, N. P., Cunningham, A. L., Harman, A. N., &
Patrick, E. (2019). Mass Cytometry Imaging for the Study of Human
Diseases---Applications and Data Analysis Strategies. *Frontiers in
Immunology*, *10*, 2657. <https://doi.org/10.3389/fimmu.2019.02657>
:::

::: {#ref-P7_baranski_maui_2021 .csl-entry}
Baranski, A., Milo, I., Greenbaum, S., Oliveria, J.-P., Mrdjen, D.,
Angelo, M., & Keren, L. (2021). MAUI (MBI Analysis User Interface)---An
image processing pipeline for Multiplexed Mass Based Imaging. *PLOS
Computational Biology*, *17*(4), e1008887.
<https://doi.org/10.1371/journal.pcbi.1008887>
:::

::: {#ref-B82_P16_ilastik2019 .csl-entry}
Berg, S., Kutra, D., Kroeger, T., Straehle, C. N., Kausler, B. X.,
Haubold, C., Schiegg, M., Ales, J., Beier, T., Rudy, M., Eren, K.,
Cervantes, J. I., Xu, B., Beuttenmueller, F., Wolny, A., Zhang, C.,
Koethe, U., Hamprecht, F. A., & Kreshuk, A. (2019). Ilastik: Interactive
machine learning for (bio)image analysis. *Nature Methods*, *16*(12),
1226--1232. <https://doi.org/10.1038/s41592-019-0582-9>
:::

::: {#ref-b130_tissueschem2022 .csl-entry}
Bhate, S. S., Barlow, G. L., Schürch, C. M., & Nolan, G. P. (2022).
Tissue schematics map the specialization of immune tissue motifs and
their appropriation by tumors. *Cell Systems*, *13*(2), 109--130.e6.
<https://doi.org/10.1016/j.cels.2021.09.012>
:::

::: {#ref-B121_birk_quantitative_2025 .csl-entry}
Birk, S., Bonafonte-Pardàs, I., Feriz, A. M., Boxall, A., Agirre, E.,
Memi, F., Maguza, A., Yadav, A., Armingol, E., Fan, R., Castelo-Branco,
G., Theis, F. J., Bayraktar, O. A., Talavera-López, C., & Lotfollahi, M.
(2025). Quantitative characterization of cell niches in spatially
resolved omics data. *Nature Genetics*, *57*(4), 897--909.
<https://doi.org/10.1038/s41588-025-02120-6>
:::

::: {#ref-B113_blondel2008 .csl-entry}
Blondel, V. D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E. (2008).
Fast unfolding of communities in large networks. *Journal of Statistical
Mechanics: Theory and Experiment*, *2008*(10), P10008.
<https://doi.org/10.1088/1742-5468/2008/10/P10008>
:::

::: {#ref-B146_nichenet2020 .csl-entry}
Browaeys, R., Saelens, W., & Saeys, Y. (2020). NicheNet: Modeling
intercellular communication by linking ligands to target genes. *Nature
Methods*, *17*(2), 159--162. <https://doi.org/10.1038/s41592-019-0667-5>
:::

::: {#ref-B144_opttransp2023 .csl-entry}
Cang, Z., Zhao, Y., Almet, A. A., Stabell, A., Ramos, R., Plikus, M. V.,
Atwood, S. X., & Nie, Q. (2023). Screening cell--cell communication in
spatial transcriptomics via collective optimal transport. *Nature
Methods*, *20*(2). <https://doi.org/10.1038/s41592-022-01728-4>
:::

::: {#ref-S7_cellprofiler2006 .csl-entry}
Carpenter, A. E., Jones, T. R., Lamprecht, M. R., Clarke, C., Kang, I.
H., Friman, O., Guertin, D. A., Chang, J. H., Lindquist, R. A., Moffat,
J., Golland, P., & Sabatini, D. M. (2006). CellProfiler: Image analysis
software for identifying and quantifying cell phenotypes. *Genome
Biology*, *7*(10), R100. <https://doi.org/10.1186/gb-2006-7-10-r100>
:::

::: {#ref-P8_chevrier_compensation_2018 .csl-entry}
Chevrier, S., Crowell, H. L., Zanotelli, V. R. T., Engler, S., Robinson,
M. D., & Bodenmiller, B. (2018). Compensation of Signal Spillover in
Suspension and Imaging Mass Cytometry. *Cell Systems*, *6*(5),
612--620.e5. <https://doi.org/10.1016/j.cels.2018.02.010>
:::

::: {#ref-B104_defard2024 .csl-entry}
Defard, T., Laporte, H., Ayan, M., Soulier, J., Curras-Alonso, S.,
Weber, C., Massip, F., Londoño-Vallejo, J.-A., Fouillade, C., Mueller,
F., & Walter, T. (2024). A point cloud segmentation framework for
image-based spatial transcriptomics. *Communications Biology*, *7*(1),
823. <https://doi.org/10.1038/s42003-024-06480-3>
:::

::: {#ref-B139_liana2024 .csl-entry}
Dimitrov, D., Schäfer, P. S. L., Farr, E., Rodriguez-Mier, P.,
Lobentanzer, S., Badia-i-Mompel, P., Dugourd, A., Tanevski, J., Ramirez
Flores, R. O., & Saez-Rodriguez, J. (2024). LIANA+ provides an
all-in-one framework for cell--cell communication inference. *Nature
Cell Biology*, *26*(9). <https://doi.org/10.1038/s41556-024-01469-w>
:::

::: {#ref-B116_dominguez2022 .csl-entry}
Domínguez Conde, C., Xu, C., Jarvis, L. B., Rainbow, D. B., Wells, S.
B., Gomes, T., Howlett, S. K., Suchanek, O., Polanski, K., King, H. W.,
Mamanova, L., Huang, N., Szabo, P. A., Richardson, L., Bolt, L.,
Fasouli, E. S., Mahbubani, K. T., Prete, M., Tuck, L., ... Teichmann, S.
A. (2022). Cross-tissue immune cell analysis reveals tissue-specific
features in humans. *Science*, *376*(6594), eabl5197.
<https://doi.org/10.1126/science.abl5197>
:::

::: {#ref-B126_dong2022 .csl-entry}
Dong, K., & Zhang, S. (2022). Deciphering spatial domains from spatially
resolved transcriptomics with an adaptive graph attention auto-encoder.
*Nature Communications*, *13*(1), 1739.
<https://doi.org/10.1038/s41467-022-29439-6>
:::

::: {#ref-B174_giotto2021 .csl-entry}
Dries, R., Zhu, Q., Dong, R., Eng, C.-H. L., Li, H., Liu, K., Fu, Y.,
Zhao, T., Sarkar, A., Bao, F., George, R. E., Pierson, N., Cai, L., &
Yuan, G.-C. (2021). Giotto: A toolbox for integrative analysis and
visualization of spatial expression data. *Genome Biology*, *22*(1), 78.
<https://doi.org/10.1186/s13059-021-02286-2>
:::

::: {#ref-B133_edsgard2018 .csl-entry}
Edsgärd, D., Johnsson, P., & Sandberg, R. (2018). Identification of
spatial expression trends in single-cell gene expression data. *Nature
Methods*, *15*(5), 339--342. <https://doi.org/10.1038/nmeth.4634>
:::

::: {#ref-B140_fischer2023 .csl-entry}
Fischer, D. S., Schaar, A. C., & Theis, F. J. (2023). Modeling
intercellular communication in tissues using spatial graphs of cells.
*Nature Biotechnology*, *41*(3), 332--336.
<https://doi.org/10.1038/s41587-022-01467-z>
:::

::: {#ref-B103_bidcell_2024 .csl-entry}
Fu, X., Lin, Y., Lin, D. M., Mechtersheimer, D., Wang, C., Ameen, F.,
Ghazanfar, S., Patrick, E., Kim, J., & Yang, J. Y. H. (2024). BIDCell:
Biologically-informed self-supervised learning for segmentation of
subcellular spatial transcriptomics data. *Nature Communications*,
*15*(1), 509. <https://doi.org/10.1038/s41467-023-44560-w>
:::

::: {#ref-B165_fu2023 .csl-entry}
Fu, X., Patrick, E., Yang, J. Y. H., Feng, D. D., & Kim, J. (2023). Deep
multimodal graph-based network for survival prediction from highly
multiplexed images and patient variables. *Computers in Biology and
Medicine*, *154*, 106576.
<https://doi.org/10.1016/j.compbiomed.2023.106576>
:::

::: {#ref-B131_getis2010 .csl-entry}
Getis, A. (2010). Spatial Autocorrelation. In M. M. Fischer & A. Getis
(Eds.), *Handbook of Applied Spatial Analysis* (pp. 255--278). Springer
Berlin Heidelberg. <https://doi.org/10.1007/978-3-642-03647-7_14>
:::

::: {#ref-P5_B44_S2_CODEX2018 .csl-entry}
Goltsev, Y., Samusik, N., Kennedy-Darling, J., Bhate, S., Hale, M.,
Vazquez, G., Black, S., & Nolan, G. P. (2018). Deep Profiling of Mouse
Splenic Architecture with CODEX Multiplexed Imaging. *Cell*, *174*(4),
968--981.e15. <https://doi.org/10.1016/j.cell.2018.07.010>
:::

::: {#ref-S5_mesmer2022 .csl-entry}
Greenwald, N. F., Miller, G., Moen, E., Kong, A., Kagel, A., Dougherty,
T., Fullaway, C. C., McIntosh, B. J., Leow, K. X., Schwartz, M. S.,
Pavelchek, C., Cui, S., Camplisson, I., Bar-Tal, O., Singh, J., Fong,
M., Chaudhry, G., Abraham, Z., Moseley, J., ... Van Valen, D. (2022).
Whole-cell segmentation of tissue images with human-level performance
using large-scale data annotation and deep learning. *Nature
Biotechnology*, *40*(4), 555--565.
<https://doi.org/10.1038/s41587-021-01094-0>
:::

::: {#ref-B171_hao2021 .csl-entry}
Hao, Y., Hao, S., Andersen-Nissen, E., Mauck, W. M., Zheng, S., Butler,
A., Lee, M. J., Wilk, A. J., Darby, C., Zager, M., Hoffman, P.,
Stoeckius, M., Papalexi, E., Mimitou, E. P., Jain, J., Srivastava, A.,
Stuart, T., Fleming, L. M., Yeung, B., ... Satija, R. (2021). Integrated
analysis of multimodal single-cell data. *Cell*, *184*(13),
3573--3587.e29. <https://doi.org/10.1016/j.cell.2021.04.048>
:::

::: {#ref-B120_covet2025 .csl-entry}
Haviv, D., Remšík, J., Gatie, M., Snopkowski, C., Takizawa, M., Pereira,
N., Bashkin, J., Jovanovich, S., Nawy, T., Chaligne, R., Boire, A.,
Hadjantonakis, A.-K., & Pe'er, D. (2025). The covariance environment
defines cellular niches for spatial inference. *Nature Biotechnology*,
*43*(2), 269--280. <https://doi.org/10.1038/s41587-024-02193-4>
:::

::: {#ref-B101_clustermap_2021 .csl-entry}
He, Y., Tang, X., Huang, J., Ren, J., Zhou, H., Chen, K., Liu, A., Shi,
H., Lin, Z., Li, Q., Aditham, A., Ounadjela, J., Grody, E. I., Shu, J.,
Liu, J., & Wang, X. (2021). ClusterMap for multi-scale clustering
analysis of spatial gene expression. *Nature Communications*, *12*(1),
5909. <https://doi.org/10.1038/s41467-021-26044-x>
:::

::: {#ref-B112_bestPractices2023 .csl-entry}
Heumos, L., Schaar, A. C., Lance, C., Litinetskaya, A., Drost, F.,
Zappia, L., Lücken, M. D., Strobl, D. C., Henao, J., Curion, F.,
Single-cell Best Practices Consortium, Aliee, H., Ansari, M.,
Badia-i-Mompel, P., Büttner, M., Dann, E., Dimitrov, D., Dony, L.,
Frishberg, A., ... Theis, F. J. (2023). Best practices for single-cell
analysis across modalities. *Nature Reviews Genetics*, *24*(8),
550--572. <https://doi.org/10.1038/s41576-023-00586-w>
:::

::: {#ref-G10_b125_spagcn .csl-entry}
Hu, J., Li, X., Coleman, K., Schroeder, A., Ma, N., Irwin, D. J., Lee,
E. B., Shinohara, R. T., & Li, M. (2021). SpaGCN: Integrating gene
expression, spatial location and histology to identify spatial domains
and spatially variable genes by graph convolutional network. *Nature
Methods*, *18*(11), 1342--1351.
<https://doi.org/10.1038/s41592-021-01255-8>
:::

::: {#ref-B83_P17_25_ijsselsteijn2021 .csl-entry}
Ijsselsteijn, M. E., Somarakis, A., Lelieveldt, B. P. F., Höllt, T., &
De Miranda, N. F. C. C. (2021). Semi‐automated background removal limits
data loss and normalizes imaging mass cytometry data. *Cytometry Part
A*, *99*(12), 1187--1197. <https://doi.org/10.1002/cyto.a.24480>
:::

::: {#ref-B161_dialogue2022 .csl-entry}
Jerby-Arnon, L., & Regev, A. (2022). DIALOGUE maps multicellular
programs in tissue from single-cell or spatial transcriptomics data.
*Nature Biotechnology*, *40*(10), 1467--1477.
<https://doi.org/10.1038/s41587-022-01288-0>
:::

::: {#ref-B164_naronet2022 .csl-entry}
Jiménez-Sánchez, D., Ariz, M., Chang, H., Matias-Guiu, X., De Andrea, C.
E., & Ortiz-de-Solórzano, C. (2022). NaroNet: Discovery of tumor
microenvironment elements from highly multiplexed images. *Medical Image
Analysis*, *78*, 102384. <https://doi.org/10.1016/j.media.2022.102384>
:::

::: {#ref-B142_cellchat2025 .csl-entry}
Jin, S., Plikus, M. V., & Nie, Q. (2025). CellChat for systematic
analysis of cell--cell communication from single-cell transcriptomics.
*Nature Protocols*, *20*(1), 180--219.
<https://doi.org/10.1038/s41596-024-01045-4>
:::

::: {#ref-B117_symphony .csl-entry}
Kang, J. B., Nathan, A., Weinand, K., Zhang, F., Millard, N., Rumker,
L., Moody, D. B., Korsunsky, I., & Raychaudhuri, S. (2021). Efficient
and precise single-cell reference atlas mapping with Symphony. *Nature
Communications*, *12*(1), 5890.
<https://doi.org/10.1038/s41467-021-25957-x>
:::

::: {#ref-P13_keren_structured_2018 .csl-entry}
Keren, L., Bosse, M., Marquez, D., Angoshtari, R., Jain, S., Varma, S.,
Yang, S.-R., Kurian, A., Van Valen, D., West, R., Bendall, S. C., &
Angelo, M. (2018). A Structured Tumor-Immune Microenvironment in Triple
Negative Breast Cancer Revealed by Multiplexed Ion Beam Imaging. *Cell*,
*174*(6), 1373--1387.e19. <https://doi.org/10.1016/j.cell.2018.08.039>
:::

::: {#ref-P19_krull_noise2void_2019 .csl-entry}
Krull, A., Buchholz, T.-O., & Jug, F. (2019). Noise2Void - Learning
Denoising From Single Noisy Images. *2019 IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR)*, 2124--2132.
<https://doi.org/10.1109/CVPR.2019.00223>
:::

::: {#ref-B93_lee_cellseg_2022 .csl-entry}
Lee, M. Y., Bedia, J. S., Bhate, S. S., Barlow, G. L., Phillips, D.,
Fantl, W. J., Nolan, G. P., & Schürch, C. M. (2022). CellSeg: A robust,
pre-trained nucleus segmentation and pixel quantification software for
highly multiplexed fluorescence images. *BMC Bioinformatics*, *23*(1),
46. <https://doi.org/10.1186/s12859-022-04570-9>
:::

::: {#ref-B145_spatialDM2023 .csl-entry}
Li, Z., Wang, T., Liu, P., & Huang, Y. (2023). SpatialDM for rapid
identification of spatially co-expressed ligand--receptor and revealing
cell--cell communication patterns. *Nature Communications*, *14*(1),
3995. <https://doi.org/10.1038/s41467-023-39608-w>
:::

::: {#ref-G15_pixie .csl-entry}
Liu, C. C., Greenwald, N. F., Kong, A., McCaffrey, E. F., Leow, K. X.,
Mrdjen, D., Cannon, B. J., Rumberger, J. L., Varra, S. R., & Angelo, M.
(2023). Robust phenotyping of highly multiplexed tissue imaging data
using pixel-level clustering. *Nature Communications*, *14*(1).
<https://doi.org/10.1038/s41467-023-40068-5>
:::

::: {#ref-B127_graphst2023 .csl-entry}
Long, Y., Ang, K. S., Li, M., Chong, K. L. K., Sethi, R., Zhong, C., Xu,
H., Ong, Z., Sachaphibulkij, K., Chen, A., Zeng, L., Fu, H., Wu, M.,
Lim, L. H. K., Liu, L., & Chen, J. (2023). Spatially informed
clustering, integration, and deconvolution of spatial transcriptomics
with GraphST. *Nature Communications*, *14*(1), 1155.
<https://doi.org/10.1038/s41467-023-36796-3>
:::

::: {#ref-P10_lu_imc-denoise_2023 .csl-entry}
Lu, P., Oetjen, K. A., Bender, D. E., Ruzinova, M. B., Fisher, D. A. C.,
Shim, K. G., Pachynski, R. K., Brennen, W. N., Oh, S. T., Link, D. C., &
Thorek, D. L. J. (2023). IMC-Denoise: A content aware denoising pipeline
to enhance Imaging Mass Cytometry. *Nature Communications*, *14*(1),
1601. <https://doi.org/10.1038/s41467-023-37123-6>
:::

::: {#ref-B96_splinedist2020 .csl-entry}
Mandal, S., & Uhlmann, V. (2020). *SplineDist: Automated Cell
Segmentation With Spline Curves*. Bioinformatics.
<https://doi.org/10.1101/2020.10.27.357640>
:::

::: {#ref-B107_optXenium2025 .csl-entry}
Marco Salas, S., Kuemmerle, L. B., Mattsson-Langseth, C., Tismeyer, S.,
Avenel, C., Hu, T., Rehman, H., Grillo, M., Czarnewski, P., Helgadottir,
S., Tiklova, K., Andersson, A., Rafati, N., Chatzinikolaou, M., Theis,
F. J., Luecken, M. D., Wählby, C., Ishaque, N., & Nilsson, M. (2025).
Optimizing Xenium In Situ data utility by quality assessment and
best-practice analysis workflows. *Nature Methods*, *22*(4), 813--823.
<https://doi.org/10.1038/s41592-025-02617-2>
:::

::: {#ref-B175_spatialdata2025 .csl-entry}
Marconato, L., Palla, G., Yamauchi, K. A., Virshup, I., Heidari, E.,
Treis, T., Vierdag, W.-M., Toth, M., Stockhaus, S., Shrestha, R. B.,
Rombaut, B., Pollaris, L., Lehner, L., Vöhringer, H., Kats, I., Saeys,
Y., Saka, S. K., Huber, W., Gerstung, M., ... Stegle, O. (2025).
SpatialData: An open and universal data framework for spatial omics.
*Nature Methods*, *22*(1), 58--62.
<https://doi.org/10.1038/s41592-024-02212-x>
:::

::: {#ref-P26_S21_mcquin_cellprofiler_2018 .csl-entry}
McQuin, C., Goodman, A., Chernyshev, V., Kamentsky, L., Cimini, B. A.,
Karhohs, K. W., Doan, M., Ding, L., Rafelski, S. M., Thirstrup, D.,
Wiegraebe, W., Singh, S., Becker, T., Caicedo, J. C., & Carpenter, A. E.
(2018). CellProfiler 3.0: Next-generation image processing for biology.
*PLOS Biology*, *16*(7), e2005970.
<https://doi.org/10.1371/journal.pbio.2005970>
:::

::: {#ref-P9_12_milosevic_different_2023 .csl-entry}
Milosevic, V. (2023). Different approaches to Imaging Mass Cytometry
data analysis. *Bioinformatics Advances*, *3*(1), vbad046.
<https://doi.org/10.1093/bioadv/vbad046>
:::

::: {#ref-B162_tensor2024 .csl-entry}
Mitchel, J., Gordon, M. G., Perez, R. K., Biederstedt, E., Bueno, R.,
Ye, C. J., & Kharchenko, P. V. (2024). Coordinated, multicellular
patterns of transcriptional variation that stratify patient cohorts are
revealed by tensor decomposition. *Nature Biotechnology*.
<https://doi.org/10.1038/s41587-024-02411-z>
:::

::: {#ref-B51_museumST_2022 .csl-entry}
Moses, L., & Pachter, L. (2022). Museum of spatial transcriptomics.
*Nature Methods*, *19*(5), 534--546.
<https://doi.org/10.1038/s41592-022-01409-2>
:::

::: {#ref-B155_collectri2023 .csl-entry}
Müller-Dott, S., Tsirvouli, E., Vazquez, M., Ramirez Flores, R. O.,
Badia-i-Mompel, P., Fallegger, R., Türei, D., Lægreid, A., &
Saez-Rodriguez, J. (2023). Expanding the coverage of regulons from
high-confidence prior knowledge for accurate estimation of transcription
factor activities. *Nucleic Acids Research*, *51*(20), 10934--10949.
<https://doi.org/10.1093/nar/gkad841>
:::

::: {#ref-S16_cellpose2 .csl-entry}
Pachitariu, M., & Stringer, C. (2022). Cellpose 2.0: How to train your
own model. *Nature Methods*, *19*(12), 1634--1641.
<https://doi.org/10.1038/s41592-022-01663-4>
:::

::: {#ref-G7_squidpy .csl-entry}
Palla, G., Spitzer, H., Klein, M., Fischer, D., Schaar, A. C.,
Kuemmerle, L. B., Rybakov, S., Ibarra, I. L., Holmberg, O., Virshup, I.,
Lotfollahi, M., Richter, S., & Theis, F. J. (2022). Squidpy: A scalable
framework for spatial omics analysis. *Nature Methods*, *19*(2),
171--178. <https://doi.org/10.1038/s41592-021-01358-2>
:::

::: {#ref-G9_ssam .csl-entry}
Park, J., Choi, W., Tiesmeyer, S., Long, B., Borm, L. E., Garren, E.,
Nguyen, T. N., Tasic, B., Codeluppi, S., Graf, T., Schlesner, M.,
Stegle, O., Eils, R., & Ishaque, N. (2021). Cell segmentation-free
inference of cell types from in situ transcriptomics data. *Nature
Communications*, *12*(1), 3545.
<https://doi.org/10.1038/s41467-021-23807-4>
:::

::: {#ref-B105_spage2vec2021 .csl-entry}
Partel, G., & Wählby, C. (2021). Spage2vec: Unsupervised representation
of localized spatial gene expression signatures. *The FEBS Journal*,
*288*(6), 1859--1870. <https://doi.org/10.1111/febs.15572>
:::

::: {#ref-B169_perezGuide2024 .csl-entry}
Perez-Lopez, R., Ghaffari Laleh, N., Mahmood, F., & Kather, J. N.
(2024). A guide to artificial intelligence for cancer researchers.
*Nature Reviews Cancer*, *24*(6), 427--441.
<https://doi.org/10.1038/s41568-024-00694-7>
:::

::: {#ref-G8_baysor .csl-entry}
Petukhov, V., Xu, R. J., Soldatov, R. A., Cadinu, P., Khodosevich, K.,
Moffitt, J. R., & Kharchenko, P. V. (2022). Cell segmentation in
imaging-based spatial transcriptomics. *Nature Biotechnology*, *40*(3),
345--354. <https://doi.org/10.1038/s41587-021-01044-w>
:::

::: {#ref-B124_pham2023 .csl-entry}
Pham, D., Tan, X., Balderson, B., Xu, J., Grice, L. F., Yoon, S.,
Willis, E. F., Tran, M., Lam, P. Y., Raghubar, A., Kalita-de Croft, P.,
Lakhani, S., Vukovic, J., Ruitenberg, M. J., & Nguyen, Q. H. (2023).
Robust mapping of spatiotemporal trajectories and cell--cell
interactions in healthy and diseased tissues. *Nature Communications*,
*14*(1), 7739. <https://doi.org/10.1038/s41467-023-43120-6>
:::

::: {#ref-B100_prabhakaran_sparcle_2022 .csl-entry}
Prabhakaran, S. (2022). Sparcle: Assigning transcripts to cells in
multiplexed images. *Bioinformatics Advances*, *2*(1), vbac048.
<https://doi.org/10.1093/bioadv/vbac048>
:::

::: {#ref-B106_qian_probabilistic_2020 .csl-entry}
Qian, X., Harris, K. D., Hauling, T., Nicoloutsopoulos, D.,
Muñoz-Manchado, A. B., Skene, N., Hjerling-Leffler, J., & Nilsson, M.
(2020). Probabilistic cell typing enables fine mapping of closely
related cell types in situ. *Nature Methods*, *17*(1), 101--106.
<https://doi.org/10.1038/s41592-019-0631-4>
:::

::: {#ref-B111_flowSOMprotocol2021 .csl-entry}
Quintelier, K., Couckuyt, A., Emmaneel, A., Aerts, J., Saeys, Y., & Van
Gassen, S. (2021). Analyzing high-dimensional cytometry data using
FlowSOM. *Nature Protocols*, *16*(8), 3775--3801.
<https://doi.org/10.1038/s41596-021-00550-0>
:::

::: {#ref-B159_MOFAcell2023 .csl-entry}
Ramirez Flores, R. O., Lanzer, J. D., Dimitrov, D., Velten, B., &
Saez-Rodriguez, J. (2023). Multicellular factor analysis of single-cell
data for a tissue-centric understanding of disease. *eLife*, *12*,
e93161. <https://doi.org/10.7554/eLife.93161>
:::

::: {#ref-B81_P14_rendeiro2021 .csl-entry}
Rendeiro, A. F., Ravichandran, H., Bram, Y., Chandar, V., Kim, J.,
Meydan, C., Park, J., Foox, J., Hether, T., Warren, S., Kim, Y., Reeves,
J., Salvatore, S., Mason, C. E., Swanson, E. C., Borczuk, A. C.,
Elemento, O., & Schwartz, R. E. (2021). The spatial landscape of lung
pathology during COVID-19 progression. *Nature*, *593*(7860), 564--569.
<https://doi.org/10.1038/s41586-021-03475-6>
:::

::: {#ref-B166_risom2022 .csl-entry}
Risom, T., Glass, D. R., Averbukh, I., Liu, C. C., Baranski, A., Kagel,
A., McCaffrey, E. F., Greenwald, N. F., Rivero-Gutiérrez, B., Strand, S.
H., Varma, S., Kong, A., Keren, L., Srivastava, S., Zhu, C., Khair, Z.,
Veis, D. J., Deschryver, K., Vennam, S., ... Angelo, M. (2022).
Transition to invasive breast cancer is associated with progressive
changes in the structure and composition of tumor stroma. *Cell*,
*185*(2), 299--310.e18. <https://doi.org/10.1016/j.cell.2021.12.023>
:::

::: {#ref-B177_julia2023 .csl-entry}
Roesch, E., Greener, J. G., MacLean, A. L., Nassar, H., Rackauckas, C.,
Holy, T. E., & Stumpf, M. P. H. (2023). Julia for biologists. *Nature
Methods*, *20*(5), 655--664.
<https://doi.org/10.1038/s41592-023-01832-z>
:::

::: {#ref-B118_histocat_2017 .csl-entry}
Schapiro, D., Jackson, H. W., Raghuraman, S., Fischer, J. R., Zanotelli,
V. R. T., Schulz, D., Giesen, C., Catena, R., Varga, Z., & Bodenmiller,
B. (2017). [histoCAT]{.nocase}: Analysis of cell phenotypes and
interactions in multiplex image cytometry data. *Nature Methods*,
*14*(9), 873--876. <https://doi.org/10.1038/nmeth.4391>
:::

::: {#ref-B94_mcmicro2022 .csl-entry}
Schapiro, D., Sokolov, A., Yapp, C., Chen, Y.-A., Muhlich, J. L., Hess,
J., Creason, A. L., Nirmal, A. J., Baker, G. J., Nariya, M. K., Lin,
J.-R., Maliga, Z., Jacobson, C. A., Hodgman, M. W., Ruokonen, J., Farhi,
S. L., Abbondanza, D., McKinley, E. T., Persson, D., ... Sorger, P. K.
(2022). MCMICRO: A scalable, modular image-processing pipeline for
multiplexed tissue imaging. *Nature Methods*, *19*(3), 311--315.
<https://doi.org/10.1038/s41592-021-01308-y>
:::

::: {#ref-B88_schindelin_fiji_2012 .csl-entry}
Schindelin, J., Arganda-Carreras, I., Frise, E., Kaynig, V., Longair,
M., Pietzsch, T., Preibisch, S., Rueden, C., Saalfeld, S., Schmid, B.,
Tinevez, J.-Y., White, D. J., Hartenstein, V., Eliceiri, K., Tomancak,
P., & Cardona, A. (2012). Fiji: An open-source platform for
biological-image analysis. *Nature Methods*, *9*(7), 676--682.
<https://doi.org/10.1038/nmeth.2019>
:::

::: {#ref-B97_stardist2018 .csl-entry}
Schmidt, U., Weigert, M., Broaddus, C., & Myers, G. (2018). *Cell
Detection with Star-convex Polygons*.
<https://doi.org/10.48550/ARXIV.1806.03535>
:::

::: {#ref-B154_progeny2018 .csl-entry}
Schubert, M., Klinger, B., Klünemann, M., Sieber, A., Uhlitz, F., Sauer,
S., Garnett, M. J., Blüthgen, N., & Saez-Rodriguez, J. (2018).
Perturbation-response genes reveal signaling footprints in cancer gene
expression. *Nature Communications*, *9*(1), 20.
<https://doi.org/10.1038/s41467-017-02391-6>
:::

::: {#ref-B119_schurch_coordinated_2020 .csl-entry}
Schürch, C. M., Bhate, S. S., Barlow, G. L., Phillips, D. J., Noti, L.,
Zlobec, I., Chu, P., Black, S., Demeter, J., McIlwain, D. R., Kinoshita,
S., Samusik, N., Goltsev, Y., & Nolan, G. P. (2020). Coordinated
Cellular Neighborhoods Orchestrate Antitumoral Immunity at the
Colorectal Cancer Invasive Front. *Cell*, *182*(5), 1341--1359.e19.
<https://doi.org/10.1016/j.cell.2020.07.005>
:::

::: {#ref-B143_spatalk2022 .csl-entry}
Shao, X., Li, C., Yang, H., Lu, X., Liao, J., Qian, J., Wang, K., Cheng,
J., Yang, P., Chen, H., Xu, X., & Fan, X. (2022). Knowledge-graph-based
cell-cell communication inference for spatially resolved transcriptomic
data with SpaTalk. *Nature Communications*, *13*(1), 4429.
<https://doi.org/10.1038/s41467-022-32111-8>
:::

::: {#ref-B102_ficture_2024 .csl-entry}
Si, Y., Lee, C., Hwang, Y., Yun, J. H., Cheng, W., Cho, C.-S., Quiros,
M., Nusrat, A., Zhang, W., Jun, G., Zöllner, S., Lee, J. H., & Kang, H.
M. (2024). FICTURE: Scalable segmentation-free analysis of
submicron-resolution spatial transcriptomics. *Nature Methods*,
*21*(10), 1843--1854. <https://doi.org/10.1038/s41592-024-02415-2>
:::

::: {#ref-B_banksy_2024 .csl-entry}
Singhal, V., Chou, N., Lee, J., Yue, Y., Liu, J., Chock, W. K., Lin, L.,
Chang, Y.-C., Teo, E. M. L., Aow, J., Lee, H. K., Chen, K. H., &
Prabhakar, S. (2024). BANKSY unifies cell typing and tissue domain
segmentation for scalable spatial omics data analysis. *Nature
Genetics*, *56*(3), 431--441.
<https://doi.org/10.1038/s41588-024-01664-3>
:::

::: {#ref-g17_napari .csl-entry}
Sofroniew, N., Lambert, T., Bokota, G., Nunez-Iglesias, J., Sobolewski,
P., Sweet, A., Gaifas, L., Evans, K., Burt, A., Doncila Pop, D.,
Yamauchi, K., Weber Mendonça, M., Buckley, G., Vierdag, W.-M., Royer,
L., Can Solak, A., Harrington, K. I. S., Ahlers, J., Althviz Moré, D.,
... Zhao, R. (2025). *Napari: A multi-dimensional image viewer for
Python*. Zenodo. <https://doi.org/10.5281/ZENODO.3555620>
:::

::: {#ref-S6_cellpose2021 .csl-entry}
Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021).
Cellpose: A generalist algorithm for cellular segmentation. *Nature
Methods*, *18*(1), 100--106.
<https://doi.org/10.1038/s41592-020-01018-x>
:::

::: {#ref-B134_sun2020 .csl-entry}
Sun, S., Zhu, J., & Zhou, X. (2020). Statistical analysis of spatial
expression patterns for spatially resolved transcriptomic studies.
*Nature Methods*, *17*(2), 193--200.
<https://doi.org/10.1038/s41592-019-0701-7>
:::

::: {#ref-B132_G12_spatialDE2018 .csl-entry}
Svensson, V., Teichmann, S. A., & Stegle, O. (2018). SpatialDE:
Identification of spatially variable genes. *Nature Methods*, *15*(5),
343--346. <https://doi.org/10.1038/nmeth.4636>
:::

::: {#ref-B152_misty2022 .csl-entry}
Tanevski, J., Flores, R. O. R., Gabor, A., Schapiro, D., &
Saez-Rodriguez, J. (2022). Explainable multiview framework for
dissecting spatial relationships from highly multiplexed data. *Genome
Biology*, *23*(1). <https://doi.org/10.1186/s13059-022-02663-5>
:::

::: {#ref-B163_NNMF2023 .csl-entry}
Townes, F. W., & Engelhardt, B. E. (2023). Nonnegative spatial
factorization applied to spatial genomics. *Nature Methods*, *20*(2),
229--238. <https://doi.org/10.1038/s41592-022-01687-w>
:::

::: {#ref-B115_louvain_leiden2019 .csl-entry}
Traag, V. A., Waltman, L., & Van Eck, N. J. (2019). From Louvain to
Leiden: Guaranteeing well-connected communities. *Scientific Reports*,
*9*(1). <https://doi.org/10.1038/s41598-019-41695-z>
:::

::: {#ref-B141_cellphonedb_2025 .csl-entry}
Troulé, K., Petryszak, R., Cakir, B., Cranley, J., Harasty, A., Prete,
M., Tuong, Z. K., Teichmann, S. A., Garcia-Alonso, L., & Vento-Tormo, R.
(2025). CellPhoneDB v5: Inferring cell--cell communication from
single-cell multiomics data. *Nature Protocols*.
<https://doi.org/10.1038/s41596-024-01137-1>
:::

::: {#ref-B150_omnipath2016 .csl-entry}
Türei, D., Korcsmáros, T., & Saez-Rodriguez, J. (2016). OmniPath:
Guidelines and gateway for literature-curated signaling pathway
resources. *Nature Methods*, *13*(12), 966--967.
<https://doi.org/10.1038/nmeth.4077>
:::

::: {#ref-B170_review2024 .csl-entry}
Unger, M., & Kather, J. N. (2024). A systematic analysis of deep
learning in genomics and histopathology for precision oncology. *BMC
Medical Genomics*, *17*(1), 48.
<https://doi.org/10.1186/s12920-024-01796-9>
:::

::: {#ref-S19_flowsom2015 .csl-entry}
Van Gassen, S., Callebaut, B., Van Helden, M. J., Lambrecht, B. N.,
Demeester, P., Dhaene, T., & Saeys, Y. (2015). FlowSOM: Using
self‐organizing maps for visualization and interpretation of cytometry
data. *CytometryPartA*, *87*(7), 636--645.
<https://doi.org/10.1002/cyto.a.22625>
:::

::: {#ref-S17_imacyte .csl-entry}
Van Unen, V., Höllt, T., Pezzotti, N., Li, N., Reinders, M. J. T.,
Eisemann, E., Koning, F., Vilanova, A., & Lelieveldt, B. P. F. (2017).
Visual analysis of mass cytometry data by hierarchical stochastic
neighbour embedding reveals rare cell types. *Nature Communications*,
*8*(1), 1740. <https://doi.org/10.1038/s41467-017-01689-9>
:::

::: {#ref-B160_mefisto2022 .csl-entry}
Velten, B., Braunger, J. M., Argelaguet, R., Arnol, D., Wirbel, J.,
Bredikhin, D., Zeller, G., & Stegle, O. (2022). Identifying temporal and
spatial patterns of variation from multimodal data using MEFISTO.
*Nature Methods*, *19*(2), 179--186.
<https://doi.org/10.1038/s41592-021-01343-9>
:::

::: {#ref-B178_scverse2023 .csl-entry}
Virshup, I., Bredikhin, D., Heumos, L., Palla, G., Sturm, G., Gayoso,
A., Kats, I., Koutrouli, M., Scverse Community, Angerer, P., Bergen, V.,
Boyeau, P., Büttner, M., Eraslan, G., Fischer, D., Frank, M., Hong, J.,
Klein, M., Lange, M., ... Theis, F. J. (2023). The scverse project
provides a computational ecosystem for single-cell omics data analysis.
*Nature Biotechnology*, *41*(5).
<https://doi.org/10.1038/s41587-023-01733-8>
:::

::: {#ref-g25_anndata .csl-entry}
Virshup, I., Rybakov, S., Theis, F. J., Angerer, P., & Wolf, F. A.
(2024). Anndata: Access and store annotated datamatrices. *Journal of
Open Source Software*, *9*(101), 4371.
<https://doi.org/10.21105/joss.04371>
:::

::: {#ref-B89_zerocostDL4MIC .csl-entry}
Von Chamier, L., Laine, R. F., Jukkala, J., Spahn, C., Krentzel, D.,
Nehme, E., Lerche, M., Hernández-Pérez, S., Mattila, P. K., Karinou, E.,
Holden, S., Solak, A. C., Krull, A., Buchholz, T.-O., Jones, M. L.,
Royer, L. A., Leterrier, C., Shechtman, Y., Jug, F., ... Henriques, R.
(2021). Democratising deep learning for microscopy with ZeroCostDL4Mic.
*Nature Communications*, *12*(1), 2276.
<https://doi.org/10.1038/s41467-021-22518-0>
:::

::: {#ref-B149_walker2022 .csl-entry}
Walker, B. L., Cang, Z., Ren, H., Bourgain-Chang, E., & Nie, Q. (2022).
Deciphering tissue structure and function using spatial transcriptomics.
*Communications Biology*, *5*(1), 220.
<https://doi.org/10.1038/s42003-022-03175-5>
:::

::: {#ref-b129_nest2023 .csl-entry}
Walker, B. L., & Nie, Q. (2023). NeST: Nested hierarchical structure
identification in spatial transcriptomic data. *Nature Communications*,
*14*(1). <https://doi.org/10.1038/s41467-023-42343-x>
:::

::: {#ref-B114_waltman2013 .csl-entry}
Waltman, L., & Van Eck, N. J. (2013). A smart local moving algorithm for
large-scale modularity-based community detection. *The European Physical
Journal B*, *86*(11), 471. <https://doi.org/10.1140/epjb/e2013-40829-0>
:::

::: {#ref-P11_wang_multiplexed_2019 .csl-entry}
Wang, Y. J., Traum, D., Schug, J., Gao, L., Liu, C., Atkinson, M. A.,
Powers, A. C., Feldman, M. D., Naji, A., Chang, K.-M., & Kaestner, K. H.
(2019). Multiplexed In Situ Imaging Mass Cytometry Analysis of the Human
Endocrine Pancreas and Immune System in Type 1 Diabetes. *Cell
Metabolism*, *29*(3), 769--783.e4.
<https://doi.org/10.1016/j.cmet.2019.01.003>
:::

::: {#ref-B168_wang2022 .csl-entry}
Wang, Y., Wang, Y. G., Hu, C., Li, M., Fan, Y., Otter, N., Sam, I., Gou,
H., Hu, Y., Kwok, T., Zalcberg, J., Boussioutas, A., Daly, R. J.,
Montúfar, G., Liò, P., Xu, D., Webb, G. I., & Song, J. (2022). Cell
graph neural networks enable the precise prediction of patient survival
in gastric cancer. *Npj Precision Oncology*, *6*(1), 45.
<https://doi.org/10.1038/s41698-022-00285-5>
:::

::: {#ref-B79_P15_S20_steinbock2023 .csl-entry}
Windhager, J., Zanotelli, V. R. T., Schulz, D., Meyer, L., Daniel, M.,
Bodenmiller, B., & Eling, N. (2023). An end-to-end workflow for
multiplexed image processing and analysis. *Nature Protocols*, *18*(11),
3565--3613. <https://doi.org/10.1038/s41596-023-00881-0>
:::

::: {#ref-G6_scanpy_2018 .csl-entry}
Wolf, F. A., Angerer, P., & Theis, F. J. (2018). SCANPY: Large-scale
single-cell gene expression data analysis. *Genome Biology*, *19*(1),
15. <https://doi.org/10.1186/s13059-017-1382-0>
:::

::: {#ref-B147_gcng2020 .csl-entry}
Yuan, Y., & Bar-Joseph, Z. (2020). GCNG: Graph convolutional networks
for inferring gene interaction from spatial transcriptomics data.
*Genome Biology*, *21*(1). <https://doi.org/10.1186/s13059-020-02214-w>
:::

::: {#ref-B137_scGCO2022 .csl-entry}
Zhang, K., Feng, W., & Wang, P. (2022). Identification of spatially
variable genes with graph cuts. *Nature Communications*, *13*(1), 5488.
<https://doi.org/10.1038/s41467-022-33182-3>
:::

::: {#ref-B122_bayespace2021 .csl-entry}
Zhao, E., Stone, M. R., Ren, X., Guenthoer, J., Smythe, K. S., Pulliam,
T., Williams, S. R., Uytingco, C. R., Taylor, S. E. B., Nghiem, P.,
Bielas, J. H., & Gottardo, R. (2021). Spatial transcriptomics at subspot
resolution with BayesSpace. *Nature Biotechnology*, *39*(11),
1375--1384. <https://doi.org/10.1038/s41587-021-00935-2>
:::

::: {#ref-B135_sparkX2021 .csl-entry}
Zhu, J., Sun, S., & Zhou, X. (2021). SPARK-X: Non-parametric modeling
enables scalable and robust detection of spatial expression patterns for
large spatial transcriptomic studies. *Genome Biology*, *22*(1), 184.
<https://doi.org/10.1186/s13059-021-02404-0>
:::

::: {#ref-B148_zhu_mapping_2024 .csl-entry}
Zhu, J., Wang, Y., Chang, W. Y., Malewska, A., Napolitano, F., Gahan, J.
C., Unni, N., Zhao, M., Yuan, R., Wu, F., Yue, L., Guo, L., Zhao, Z.,
Chen, D. Z., Hannan, R., Zhang, S., Xiao, G., Mu, P., Hanker, A. B., ...
Wang, T. (2024). Mapping cellular interactions from spatially resolved
transcriptomics data. *Nature Methods*, *21*(10), 1830--1842.
<https://doi.org/10.1038/s41592-024-02408-1>
:::

::: {#ref-B123_zhu2018 .csl-entry}
Zhu, Q., Shah, S., Dries, R., Cai, L., & Yuan, G.-C. (2018).
Identification of spatially associated subpopulations by combining
[scRNAseq]{.nocase} and sequential fluorescence in situ hybridization
data. *Nature Biotechnology*, *36*(12), 1183--1190.
<https://doi.org/10.1038/nbt.4260>
:::
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
