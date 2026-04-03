# scFactor-Ontology

`scfo` is a Python package for building, querying, projecting, and visualizing **factor-centric ontology resources** from single-cell transcriptomic analyses.

The package is designed for workflows in which a matrix of **gene-by-factor loadings** is treated as a reusable biological reference and augmented with additional annotation modalities such as:

- transcription factor regulons
- ligand–receptor communication signatures
- curated pathway or gene-set annotations
- projection scores in external single-cell datasets

## What can scFO do?

With `scfo`, you can:

- **build an ontology** from factor loadings plus optional gene set, regulon and ligand–receptor inputs
- **inspect and visualize** factor weights, modality enrichments, and feature loadings
- **project ontology factors** onto external `AnnData` objects
- **test ontology scores** between biological groups at single-cell or pseudobulk level
- **query the ontology** with either:
  - a curated unordered gene set
  - a weighted genome-wide signature
- **export and reload** ontology objects through standardized Excel workbooks

## Installation

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

## Quickstart

```python
from scfo import make_ontology, project_ontology

ontology = make_ontology(
    factor_loadings=factor_loadings.fillna(0),
    regulon_loadings=regulon_loadings.fillna(0),
    liana=liana,
    n_iter=1000,
)

project_ontology(
    adata=adata_external,
    ontology=ontology,
    annotation_key="final_annotation_fine",
    score_key_added="ontology_scores",
    pval_key_added="ontology_pvals",
    method="permutation",
    n_iter=1000,
    inplace=True,
)
```

## Core concepts

### Factor-centric ontology object

`make_ontology()` returns a factor-centric `MuData` object:

- `ontology.obs` = factor metadata
- `ontology.obsm["weights"]` = global factor weight matrix
- `ontology.mod[...]` = modality-specific enrichment matrices

### Expected input formats

- `factor_loadings`: **genes x factors**
- `regulon_loadings`: **genes x regulons**
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

- **Vignette**: end-to-end workflow and examples
- **API**: function reference with parameters and return values

## Notes on signatures

`signature_enrichment()` supports two query modes:

- **unordered gene list** → scored with **GSEA**
- **weighted `pandas.Series` or one-column `DataFrame`** → scored with the **permutation test**

The same function can query:

- global factor weights
- any ontology modality
- or both together using `search_in`

## Notes on LIANA

The package builds LIANA-derived ontology modalities from the original paired-signature logic:

- ligand-side rows: `sender_cell_type|ligand_gene`, columns: `rec by|receiver_cell_type`
- receptor-side rows: `receiver_cell_type|receptor_gene`, columns: `sent by|sender_cell_type`

During ontology assembly, each factor is enriched only against the matching lineage-specific signature block. The final `liana_ligand` and `liana_receptor` modalities therefore expose the **shared pair-wise sender/receiver columns**, while lineage-specific signature definitions are preserved in `varm`.

## Recent changes

- `signature_enrichment()` now uses `search_in` as the sole target selector
- helper functions support `cell_types=` filtering
- plotting helpers support both factor-centric and modality-centric views

