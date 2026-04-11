# scFactor-Ontology

`scfo` is a Python package for working with latent factors (gene x factor matrices with per-gene loadings) derived from single-cell transcriptomic analyses. 

## What is an `scfo ontology` object?
`scfo` was built as a tool to organize latent factor gene loadings and associated prior-knowledge annotations - for example, calculated using a large-scale single cell atlas - into interpretable, factor-centric ontology resources. 

The data structure of an `scfo ontology` object is based on `muon MuData`, the multi-modal extension of `AnnData`, treating factors as observations in `.obs`. This allows for easy storage of factor loadings and prior-knowledge annotation data as separate `MuData` modalities. 

Typical factor-wise enrichment score modalities stored in `scfo ontology` objects may include transcription factor regulons, ligand-receptor communication signatures, gene set enrichment analyses (e.g. MSigDB Hallmark Gene Ontology), etc. 

## What can scFO do?
The centralized, organized design of `scfo` streamlines many downstream analyses that are commonly performed to interpret latent factors. `scfo` functionality includes:
- ***de novo* calculation** of enrichment scores based on feature loadings (e.g. gene-wise regulon matrices)
- **plotting suite** for visualization of factor loadings/feature loadings/enrichment scores
- **querying** an `scfo ontology` for genes/sets of interest
- **projecting** an `scfo ontology` onto external `AnnData` objects (e.g. perturbational data).
- **exporting and reloading** 'ontology' objects as `.h5mu` or as standardized Excel workbooks

## How do I use scFO?
See **"Installation and Usage"** below for quick-start instructions, and the **"scFO Overview + General Workflow"** page (under Tutorials) for a high-level vignette.

This ReadTheDocs site also includes detailed tutorials for using specific `scfo` functionalities. First, decide between the following:
- **If you have new factors and prior knowledge loadings:** see Tutorials #1, #5, and #6 to learn how to construct your factor `scfo ontology`.
- **If you want to use a pre-existing `scfo ontology`:** see Tutorials #2, #3, #4, and #7 to learn how to use the `scfo ontology` to visualize factors/loadings (#2), add new modalities (#4), and project factor scores onto external data (#3 and #7).

## Do `scfo ontology` objects already exist that I can download?
Yes! We have generated `scfo ontology` objects for general use by the neuro-oncology community, representing cell type-specific latent factors from 1) a 3.3 million-cell atlas of human glioblastoma (GBM), and 2) a 400,000-cell atlas of human low grade gliomas (LGG). 
- Google Drive with example data: https://drive.google.com/drive/folders/1V0PoMmk5PseL3cSd1xrHr_btArUtjnhI?usp=sharing
- **To request access to example data on Google Drive, email Matt Warren at mrw2204@cumc.columbia.edu**

## Installation and Usage:

### From GitHub
```bash
pip install "scfo @ git+https://github.com/mrw2204/scFactor-Ontology.git@main"
```

### Development install
```bash
git clone https://github.com/mrw2204/scFactor-Ontology.git
cd scFactor-Ontology
pip install -e .
```

### Factor-centric ontology object
`make_ontology()` returns a factor-centric `MuData` object:
- `ontology.obs` = factor metadata
- `ontology.mod["weights"]` = global factor weight matrix
- `ontology.mod[...]` = modality-specific enrichment matrices

### Expected input formats for common modalities:
- `factor_loadings`: **genes x factors**
- `regulon_loadings`: **genes x regulons**
- `hallmark_lib_dict` and `gene_sets_dict`: **dictionaries of gene sets (unranked lists)**
- `liana`: raw long-form LIANA output
- external projections: `AnnData` with genes in `adata.var_names`

### Naming conventions
The package works best when factor names follow:
```text
Factor0|Tumor
Factor1|Tumor
Factor0|Mo_TAM
```
For lineage-specific regulons, use columns like:
```text
CEBPB(+)|Tumor
STAT3(+)|Mo_TAM
IRF1(+)|Endothelial
```

## Documentation
- **Tutorials**: end-to-end workflows and examples
- **API**: function reference with parameters and return values
