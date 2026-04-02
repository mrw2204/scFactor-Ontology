
# Single Cell Factor-Ontology (scFO)

Portable utilities for building, querying, projecting, and exporting **factor-centric ontology resources** from gene-by-factor loading matrices.

## Main functionality

- Build a factor-centric `MuData` ontology from:
  - factor loadings (`genes x factors`)
  - regulon loadings
  - LIANA ligand/receptor annotations
  - optional hallmark or other gene-set modalities
- Project ontology factor scores onto an external `AnnData`
- Run simple differential testing of projected ontology scores
- Query ontology factors or modalities with a custom gene signature
- Export and reload ontology objects as standardized Excel workbooks
- Plot top positive/negative features for a factor in any modality

## Conventions

- `factor_loadings`: **genes x factors**
- factor names: preferably `Factor0|Tumor`, `Factor1|Mo_TAM`, etc.
- lineage-specific regulon names: `TF(+)|cell_type`
- raw LIANA input: long-form DataFrame with columns:
  - `source`
  - `target`
  - `ligand_complex`
  - `receptor_complex`
  - `magnitude_rank`
  - `specificity_rank`
- LIANA signatures are built using the original pairwise sender/receiver logic (`magnitude_rank * specificity_rank` with per-block z-scoring and factor-support filtering), then matched to the corresponding factor lineage during ontology assembly.

## Install

```bash
pip install -e /path/to/ontology_tools_pkg_release
```

## Quick start

```python
from scfo import make_ontology, project_ontology

ontology = make_ontology(
    factor_loadings=factor_loadings,
    regulon_loadings=regulon_loadings,
    liana=liana,
    n_iter=1000,
)

project_ontology(
    adata=adata_external,
    ontology=ontology,
    annotation_key='final_annotation_fine',
    score_key_added='ontology_scores',
    pval_key_added='ontology_pvals',
    method='permutation',
)
```

See `docs/vignette.md` for a full end-to-end example.


## Weighted and unweighted signatures

`signature_enrichment()` accepts either:

- an unweighted gene list, e.g. `['CEBPB', 'LGALS3']`
- a weighted signature as a `pd.Series` or one-column `pd.DataFrame`, with gene names as the index and weights as the values

You can query the global factor weights and any modality in the same call by using `search_in`, for example `search_in=["weights", "regulons", "liana_ligand"]`. For factor-weight results, `cell_types` can be used to restrict the search to one or a few ontology lineages.


## Helper: retrieve factor weights as a DataFrame

```python
from scfo import factor_weights_to_df

# factor x gene
weights_df = factor_weights_to_df(ontology)

# gene x factor
weights_gene_by_factor = factor_weights_to_df(ontology, transpose=True)
```


## Recent API notes

### `signature_enrichment()`

`signature_enrichment()` now uses **only** the `search_in` argument to choose ontology targets. It accepts:

- an unordered gene list, which is scored with **GSEA**
- a weighted `pandas.Series` or one-column `pandas.DataFrame`, which is scored with the **permutation test**

It returns, for each queried target, a dictionary with:

- `results`: ranked enrichment table
- `overlap`: summary of overlap between the query genes and the queried ontology target

Example:

```python
res = signature_enrichment(
    signature=["CEBPB", "LGALS3", "JUNB"],
    ontology=ontology,
    search_in=["weights", "regulons", "liana_ligand"],
    cell_types=["Tumor", "Mo_TAM"],
)

res["weights"]["results"].head()
res["weights"]["overlap"]
```

### DataFrame helpers with `cell_types`

The following helpers now accept `cell_types=` to return only ontology subsets of interest:

- `factor_weights_to_df()`
- `modality_scores_to_df()`
- `modality_pvals_to_df()`
- `modality_feature_loadings_to_df()`
- `get_factor_scores()`

Examples:

```python
weights_tumor = factor_weights_to_df(ontology, cell_types="Tumor", transpose=True)
reg_scores_tumor = modality_scores_to_df(ontology, "regulons", cell_types="Tumor")
reg_loadings_tumor = modality_feature_loadings_to_df(ontology, "regulons", cell_types="Tumor")
```


### Plot genes or factors for a modality feature

```python
from scfo import plot_modality_feature_top_items

# top genes contributing to a regulon in Tumor
fig, ax = plot_modality_feature_top_items(
    ontology,
    modality="regulons",
    feature="CEBPB(+)",
    cell_type="Tumor",
    what="genes",
    n_pos=10,
    n_neg=10,
)

# top Tumor factors enriched for that regulon
fig, ax = plot_modality_feature_top_items(
    ontology,
    modality="regulons",
    feature="CEBPB(+)",
    cell_type="Tumor",
    what="factors",
    n_pos=10,
    n_neg=10,
)
```
