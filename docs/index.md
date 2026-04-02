
# ontology-tools

```{toctree}
:maxdepth: 2

vignette
```


## Weighted and unweighted signatures

`signature_enrichment()` accepts either:

- an unweighted gene list, e.g. `['CEBPB', 'LGALS3']`
- a weighted signature as a `pd.Series` or one-column `pd.DataFrame`, with gene names as the index and weights as the values

You can query the global factor weights and any modality in the same call by using `search_in`, for example `search_in=["weights", "regulons", "liana_ligand"]`. For factor-weight results, `cell_types` can be used to restrict the search to one or a few ontology lineages.


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
from ontology_tools import plot_modality_feature_top_items

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
