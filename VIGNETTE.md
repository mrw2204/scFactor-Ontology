
# ontology-tools vignette

This vignette shows a typical end-to-end workflow for `ontology_tools` using:

- `factor_loadings`: gene-by-factor loading matrix
- `regulon_loadings`: gene-by-regulon loading matrix
- `liana`: raw long-form LIANA output
- an external `AnnData` object for projection and downstream testing

## Installation

```bash
pip install -e /path/to/ontology_tools_pkg_release
```

## Imports

```python
from ontology_tools import (
    make_ontology,
    project_ontology,
    diff_exp_ontology,
    signature_enrichment,
    export_ontology_excel,
    load_ontology_excel,
    modality_scores_to_df,
    modality_feature_loadings_to_df,
    top_features_for_factor,
    plot_factor_top_features,
)
```

## Expected input formats

### Factor loadings

`factor_loadings` should be a **DataFrame with genes as rows and factors as columns**.

```python
factor_loadings.shape
# (n_genes, n_factors)
```

Recommended factor names:

```text
Factor0|Tumor
Factor1|Tumor
Factor0|Mo_TAM
```

### Regulon loadings

`regulon_loadings` should be a **DataFrame with genes as rows and regulons as columns**.

For lineage-specific regulons, use columns like:

```text
CEBPB(+)|Tumor
STAT3(+)|Mo_TAM
IRF1(+)|Endothelial
```

The package will:
- compute enrichment only against the matching lineage-specific regulon
- expose the final regulon modality with shared variables like `CEBPB(+)`
- store lineage-specific regulon definitions in `ontology.mod['regulons'].varm`

### LIANA input

Pass the **raw long-form LIANA table**, not precomputed LR signatures. It should contain at least:

- `source`
- `target`
- `ligand_complex`
- `receptor_complex`
- `magnitude_rank`
- `specificity_rank`

Example:

```python
liana[['source', 'target', 'ligand_complex', 'receptor_complex', 'magnitude_rank', 'specificity_rank']].head()
```

The package reproduces the original workflow used to construct LR signatures:
- `score = magnitude_rank * specificity_rank`
- factor support is computed from normalized factor weights
- interaction blocks are z-scored within each sender/receiver pair
- filtered ligand/receptor signatures are built from retained interactions

For the final ontology:
- `liana_ligand` variables are shared context names like `rec by|Tumor`, `rec by|Mo_TAM`, ...
- `liana_receptor` variables are shared context names like `sent by|Tumor`, `sent by|Mo_TAM`, ...
- Internally, LIANA signatures are first constructed in the full pairwise form (for example `Tumor|rec by|Pericyte` and `Tumor|sent by|Pericyte`) so that each factor is enriched only against the matching lineage-specific LR signatures, mirroring the original analysis logic. In the final ontology object, these are collapsed to the shared 11-context views described above, while lineage-specific signature matrices are preserved in `varm`.
- enrichment is computed only against the factor's own lineage-specific LR signatures
- lineage-specific LR definitions are stored in `varm`

## Build the ontology

```python
ontology = make_ontology(
    factor_loadings=factor_loadings.fillna(0),
    regulon_loadings=regulon_loadings.fillna(0),
    liana=liana,
    n_iter=1000,
)
```

This returns a factor-centric `MuData` object:

- `ontology.obs` = factors
- `ontology.obsm['weights']` = factor weight matrix
- `ontology.mod[...]` = annotation modalities

Typical modalities:

```python
list(ontology.mod.keys())
```

Expected output:

```text
['regulons', 'liana_ligand', 'liana_receptor']
```

## Inspect modalities

### Factor metadata

```python
ontology.obs.head()
```

### Regulon scores

```python
reg_scores = modality_scores_to_df(ontology, 'regulons')
reg_scores.head()
```

### LIANA ligand scores

```python
lig_scores = modality_scores_to_df(ontology, 'liana_ligand')
lig_scores.head()
```

This should now have shared context columns like:

```text
rec by|Tumor
rec by|Mo_TAM
rec by|Mg_TAM
...
```

### Retrieve feature-loading matrices

For regulons, lineage-specific definitions are stored separately:

```python
ontology.mod['regulons'].varm.keys()
```

For example:

```text
Tumor_regulons
Mo_TAM_regulons
...
```

To recover one as a DataFrame:

```python
tumor_regulons = modality_feature_loadings_to_df(ontology, 'regulons', key='Tumor_regulons')
tumor_regulons.head()
```

Similarly for LIANA ligand signatures:

```python
ontology.mod['liana_ligand'].varm.keys()
# e.g. Tumor_ligand_signatures, Mo_TAM_ligand_signatures, ...

tumor_lig = modality_feature_loadings_to_df(ontology, 'liana_ligand', key='Tumor_ligand_signatures')
```

## Top features for a factor

```python
pos, neg = top_features_for_factor(
    ontology,
    factor='Factor0|Tumor',
    modality='regulons',
    n_pos=10,
    n_neg=10,
)
```

Or plot them directly:

```python
fig, ax = plot_factor_top_features(
    ontology,
    factor='Factor0|Tumor',
    modality='regulons',
    n_pos=10,
    n_neg=10,
)
```

## Project ontology factors onto an external `AnnData`

### Global projection

```python
project_ontology(
    adata=adata_external,
    ontology=ontology,
    layer=None,
    annotation_key=None,
    score_key_added='ontology_scores',
    pval_key_added='ontology_pvals',
    method='permutation',
    n_iter=1000,
    inplace=True,
)
```

Projected scores are stored in:

```python
adata_external.obsm['ontology_scores']
```

### Cell-type-aware projection

If the external dataset has lineage labels that match the ontology classifications:

```python
project_ontology(
    adata=adata_external,
    ontology=ontology,
    layer='log1p' if 'log1p' in adata_external.layers else None,
    annotation_key='final_annotation_fine',
    score_key_added='ontology_scores',
    pval_key_added='ontology_pvals',
    method='permutation',
    n_iter=1000,
    inplace=True,
)
```

This is usually preferred when factors are lineage-specific.

### Fast dot-product projection

```python
project_ontology(
    adata=adata_external,
    ontology=ontology,
    annotation_key='final_annotation_fine',
    method='dot',
    inplace=True,
)
```

## Differential testing of ontology scores

### Single-cell testing

```python
test_adata, de_results = diff_exp_ontology(
    adata=adata_external,
    ontology_keys='ontology_scores',
    groupby='Status',
    method='wilcoxon',
    pseudo_bulk=False,
)
```

### Pseudobulk testing

```python
test_adata_pb, de_results_pb = diff_exp_ontology(
    adata=adata_external,
    ontology_keys='ontology_scores',
    groupby='Status',
    method='wilcoxon',
    pseudo_bulk=True,
    pseudo_bulk_by=['Patient_Study'],
    summary_metric='mean',
)
```

## Signature enrichment against ontology factors or modalities

```python
sig_results = signature_enrichment(
    signature=['CEBPB', 'LGALS3', 'JUNB', 'ANXA1', 'VEGFA'],
    ontology=ontology,
    modalities=['regulons', 'liana_ligand', 'liana_receptor'],
    pval_thresh=0.05,
)
```

Examples:

```python
sig_results['weights'].head()
sig_results['regulons'].head()
sig_results['liana_ligand'].head()
```

## Export to Excel

```python
paths = export_ontology_excel(
    ontology=ontology,
    outdir='.',
    prefix='gbm_ontology',
)
print(paths)
```

This writes:

- `gbm_ontology_scores.xlsx`
- `gbm_ontology_features.xlsx`

## Load ontology from Excel

```python
ontology2 = load_ontology_excel(
    scores_xlsx='gbm_ontology_scores.xlsx',
    features_xlsx='gbm_ontology_features.xlsx',
)
```

## End-to-end example

```python
from ontology_tools import (
    make_ontology,
    project_ontology,
    diff_exp_ontology,
    signature_enrichment,
    export_ontology_excel,
)

ontology = make_ontology(
    factor_loadings=factor_loadings.fillna(0),
    regulon_loadings=regulon_loadings.fillna(0),
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
    n_iter=1000,
    inplace=True,
)

_, de_results = diff_exp_ontology(
    adata=adata_external,
    ontology_keys='ontology_scores',
    groupby='Status',
    method='wilcoxon',
    pseudo_bulk=True,
    pseudo_bulk_by=['Patient_Study'],
    summary_metric='mean',
)

sig_results = signature_enrichment(
    signature=['CEBPB', 'LGALS3', 'JUNB', 'ANXA1', 'VEGFA'],
    ontology=ontology,
    modalities=['regulons', 'liana_ligand', 'liana_receptor'],
    pval_thresh=0.05,
)

paths = export_ontology_excel(ontology, outdir='.', prefix='gbm_ontology')
```


## Weighted and unweighted signatures

`signature_enrichment()` accepts either:

- an unweighted gene list, e.g. `['CEBPB', 'LGALS3']`
- a weighted signature as a `pd.Series` or one-column `pd.DataFrame`, with gene names as the index and weights as the values



You can query the global factor weights and any modality in the same call by using `search_in`, for example `search_in=["weights", "regulons", "liana_ligand"]`. For factor-weight results, `cell_types` can be used to restrict the search to one or a few ontology lineages.

## LIANA note

The package constructs intermediate LIANA signature matrices using the original paired formulation:

- ligand-side rows: `sender_cell_type|ligand_gene`, columns: `rec by|receiver_cell_type`
- receptor-side rows: `receiver_cell_type|receptor_gene`, columns: `sent by|sender_cell_type`

During ontology assembly, each factor is enriched only against the matching lineage-specific row block, so the final `liana_ligand` and `liana_receptor` modalities expose the shared 11 paired-context columns rather than all 121 sender/receiver combinations.


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
