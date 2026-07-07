# The Spatial Omics Technical Overview
Welcome to this theoretical text section. This technical overview specifically 
focuses on single-molecule or single-cell resolution approaches.

---
⚠️⚠️⚠️⚠️⚠️⚠️⚠️

Disclaimer: This overview is a preliminary text written in the context of my PhD research and has not yet undergone peer review. 
It summarizes my understanding of spatial omics datasets analysis.
---

If you find this information useful, please consider acknowledging its source!

---

Last updated: 2025-06-27

---

### Understanding Tissue Complexity: The Journey to Spatial Omics

Understanding the intricate organization of tissues in health and disease relies on advanced analytical
technologies. 

Historically, cancer research has been driven by genomic and transcriptomic approaches. 
These have effectively revealed key mutations, epigenetic changes, and dysregulated pathways within
various tumor types.

More recently, multi-omics strategies (including epigenomics, proteomics, and metabolomics) have
significantly expanded our understanding of tumor biology by capturing additional layers of molecular
regulation. Further revolutionizing the field, single-cell technologies have enabled the detailed 
dissection of intra-tumoral heterogeneity, identifying rare cell populations and lineage trajectories 
that bulk analyses often obscure. These advancements have even extended to single-cell proteomics and
metabolomics, offering increasingly comprehensive insights into tumor ecosystems.

Despite their immense power, a critical limitation of many of these cutting-edge omics methods is
their lack of spatial context. They tell us what molecular components or cell types are present,
but not where they are located or how they interact within the intricate tissue microenvironment.
This spatial information is essential for understanding disease progression and treatment response.

In clinical settings, spatial information is routinely obtained through methods like histopathological
staining (e.g., H&E), immunohistochemistry, and radiological imaging. 
While vital for diagnosis and clinical decision-making, these approaches typically offer 
spatial resolution at the tissue or organ level. 
They are inherently limited in molecular detail and throughput, lacking the capacity to resolve 
cellular heterogeneity or dynamic molecular interactions at a fine scale.

To overcome these limitations and integrate high-resolution molecular profiling with precise spatial
localization, spatial omics technologies have emerged. These powerful tools allow for the direct 
mapping of transcripts, proteins, and metabolites within their native tissue context. 
By preserving the original architecture of tumors and their microenvironments, spatial 
omics provides a more complete and insightful picture of biological processes.

The following chapters aim to provide a comprehensive understanding of spatial omics workflows, 
covering topics from:

* [Spatially Resolved Omics Techniques](chapter1.md)
* [Technical Overview](chapter2.md)
    * [Preprocessing](chapter2.md##Data preprocessing)
    * [Cell Segmentation](chapter2.md##Cell segmentation)
    * [Phenotyping](chapter2.md##Cell phenotyping)
    * [Neighborhood Analysis](chapter2.md##Cellular neighbourhood analysis)
    * [Domain Identification](chapter2.md##Spatial domains)
    * [Spatially Variable Genes](chapter2.md##Spatially Variable genes)
    * [Functional Analysis](chapter2.md##Functional analysis)
    * [Sample Condition Distinctions](chapter2.md##How to distinguish between sample conditions)
    * [Frameworks and tools](chapter2.md##Frameworks and tools for spatial omics analysis)
    * [Closing remarks](chapter2.md##Closing remarks)





