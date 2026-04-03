# scFactor-Ontology vignette

This vignette shows a typical end-to-end workflow for `scfo` using:

- `factor_loadings`: gene-by-factor loading matrix
- `hallmark_lib_dict` and `gene_sets_dict`: dictionaries of MSigDB Hallmark gene sets or user-defined gene sets (unranked lists)
- `regulon_loadings`: gene-by-regulon loading matrix
- `liana`: raw long-form LIANA output
- an external `AnnData` object for projection and downstream testing

## Installation

```bash
pip install "scfo @ git+https://github.com/mrw2204/scFactor-Ontology.git@main"
```

## Imports

```python
from scfo import (
    make_ontology,
    project_ontology,
    diff_exp_ontology,
    signature_enrichment,
    export_ontology_excel,
    load_ontology_excel,
    factor_weights_to_df,
    modality_scores_to_df,
    modality_feature_loadings_to_df,
    top_features_for_factor,
    plot_factor_top_features,
    plot_modality_feature_top_items,
)
```

## Expected input formats

### Factor loadings

- `factor_loadings` should be a DataFrame with genes as rows and factors as columns. Factors can be bi-directional (e.g. MOFA) or uni-directional (e.g. scHPF).
- Enrichment-based meta-annotation of these factors will be calculated based on user-chosen modalities (e.g. PySCENIC regulons, LIANA ligand-receptor scores, Hallmark/gene sets, etc.). 
- Modalities can be arbitrary lists/vectors of genes representing any source of prior knowledge.
- The package automatically uses preranked GSEA for enrichment of gene sets, and a permutation-based dot product algorithm for enrichment of vectors in factors. Empirical p-values are also returned.

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

### Preranked gene set enrichment

- Performs gene-ontology (GO)-style analysis.
- `hallmark_lib` should be a dictionary with the desired species/release for MSigDB Hallmark gene sets.
- `gene_sets` should be a dictionary of additional gene sets (lists) of interest.

For gene sets, use a dictionary like:

```text
gene_sets_dict = {
        'Mesenchymal': ['CEBPB', 'VIM', 'LGALS3', ...],
        'Proneural': ['SOX2', 'OLIG2', 'PDGFRA', ...],
        ...
    }
```

The package will:

- compute enrichment scores/p-values using pre-ranked gene set enrichment analysis (GSEA with gseapy) with each factor as a ranked signature
- assign the 'hallmarks' and 'gene_sets' modalities with variable names corresponding to keys in the passed dictionaries
- store factor-wise enrichment scores and nominal p-values from GSEA as .X of the modality

### Regulon loadings

`regulon_loadings` should be a DataFrame with genes as rows and regulons as columns.

For lineage-specific regulons, use columns like:

```text
CEBPB(+)|Tumor
STAT3(+)|Mo_TAM
IRF1(+)|Endothelial
```

The package will:

- compute enrichment of lineage-specific regulons within each factor of the corresponding lineage only.
- convert variable names to shared transcription factors such as `CEBPB(+)`
- store lineage-specific regulon definitions in `ontology.mod["regulons"].varm`

### LIANA input

Pass the raw long-form LIANA table, not precomputed signature matrices. It should contain at least:

- `source`
- `target`
- `ligand_complex`
- `receptor_complex`
- `magnitude_rank`
- `specificity_rank`

Example:

```python
liana[
    ["source", "target", "ligand_complex", "receptor_complex", "magnitude_rank", "specificity_rank"]
].head()
```

The package implements user-defined pruning of the raw LIANA output to construct LR signatures:

- scores individual LR pairs between cell_type_sender and cell_type_receiver via `score = magnitude_rank * specificity_rank`
- constructs pair-wise LR signatures representing overall (summed score) communication between sender and receiver populations (e.g. signature `rec by|Mo_TAM` containing summed scores for each `ligand|Tumor` summed across every possible `receptor|Mo_TAM`)
- Distinguishes "sending" and "receiving" activities as two separate signature sets
- LR pair loadings within signatures can be pruned with `liana_z_threshold`
- Allows optional pruning based on representation of LR pairs in factors, computed from normalized factor weights ,via `factor_z_threshold`
- Filtered ligand and receptor signatures are built from retained interactions and used as signatures for permutation-based enrichment in factors
- Answers the following question format: "Factor0|Tumor is involved in sending ligands to Mo-TAMs because of its positive `rec by|Mo_TAM` enrichment score."

For the final ontology:

- `liana_ligand` variables are shared context names like `rec by|Tumor`, `rec by|Mo_TAM`, ...
- `liana_receptor` variables are shared context names like `sent by|Tumor`, `sent by|Mo_TAM`, ...

Lineage-specific gene-wise signature matrices are preserved in `varm`.

## Build the ontology

```python
ontology = make_ontology(
    factor_loadings=factor_loadings.fillna(0),
    regulon_loadings=regulon_loadings.fillna(0),
    hallmark_lib=hallmark_lib_dict,
    gene_sets=gene_sets_dict,
    liana=liana,
    n_iter=1000,
)
```

This returns a factor-centric `MuData` object:

- `ontology.obs` = factors
- `ontology.obsm["weights"]` = factor weight matrix
- `ontology.mod[...]` = annotation modalities

Typical modalities:

```python
list(ontology.mod.keys())
# ['regulons', 'liana_ligand', 'liana_receptor', 'hallmark', 'gene_sets']
```

## Inspect the ontology

### Factor metadata

```python
ontology.obs.head()
```

### Retrieve factor weights

```python
weights_fxg = factor_weights_to_df(ontology)
weights_gxf = factor_weights_to_df(ontology, transpose=True)
weights_tumor = factor_weights_to_df(ontology, transpose=True, cell_types="Tumor")
```

### Retrieve modality scores

```python
reg_scores = modality_scores_to_df(ontology, "regulons")
lig_scores = modality_scores_to_df(ontology, "liana_ligand", cell_types="Tumor")
```

### Retrieve feature-loading matrices

```python
tumor_regulons = modality_feature_loadings_to_df(
    ontology,
    "regulons",
    key="Tumor_regulons",
)
```

Similarly for LIANA:

```python
tumor_lig = modality_feature_loadings_to_df(
    ontology,
    "liana_ligand",
    key="Tumor_ligand_signatures",
)
```

## Top features for a factor

```python
pos, neg = top_features_for_factor(
    ontology,
    factor="Factor0|Tumor",
    modality="regulons",
    n_pos=10,
    n_neg=10,
)
```

Or plot them directly:

```python
fig, ax = plot_factor_top_features(
    ontology,
    factor="Factor0|Tumor",
    modality="hallmark",
    n_pos=10,
    n_neg=10,
)
```

You can also plot factor weights directly:

```python
fig, ax = plot_factor_top_features(
    ontology,
    factor="Factor0|Tumor",
    modality="weights",
    n_pos=10,
    n_neg=10,
)
```

## Plot genes or factors for a modality feature

Top genes contributing to a modality feature:

```python
fig, ax = plot_modality_feature_top_items(
    ontology,
    modality="regulons",
    feature="CEBPB(+)",
    cell_type="Tumor",
    what="genes",
    n_pos=10,
    n_neg=10,
)
```

Top factors enriched in a modality feature:

```python
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

## Project ontology factors onto an external `AnnData`

- Factors can be projected onto external expression data in AnnData format. Projection scores are calculated as the dot product `expression @ loadings`, using the intersection of gene names.
- This can be done in "Global" mode (every factor projected onto every cell), or "Cell-type-aware" mode (restricting comparisons to cells x factors with matching annotations).
- If `annotation_key` is only partially overlapping with annotation categories in `ontology`, then projections are only calculated for the intersection of annotations.

### Global projection

```python
project_ontology(
    adata=adata_external,
    ontology=ontology,
    layer=None,
    annotation_key=None,
    score_key_added="ontology_scores",
    pval_key_added="ontology_pvals",
    method="permutation",
    n_iter=1000,
    inplace=True,
)
```

### Cell-type-aware projection

```python
project_ontology(
    adata=adata_external,
    ontology=ontology,
    layer="log1p" if "log1p" in adata_external.layers else None,
    annotation_key="final_annotation_fine",
    score_key_added="ontology_scores",
    pval_key_added="ontology_pvals",
    method="permutation",
    n_iter=1000,
    inplace=True,
)
```

If projection is done in cell-type-aware mode, the resulting dataframe in `.obsm` contains NaN values for non-tested cell type x factor combinations. To collapse to a more convient format (`n_cells` x `n_factors`), use the following:

```python
collapsed = scfo.collapse_projected_ontology_scores(
    adata=adata_external,
    score_key="ontology_scores",
    output_key="ontology_scores_collapsed",
    agg="sum",
    store_sparse=False,
)
```


### Fast dot-product projection

```python
project_ontology(
    adata=adata_external,
    ontology=ontology,
    annotation_key="final_annotation_fine",
    method="dot",
    inplace=True,
)
```

## Differential testing of ontology scores

### Single-cell testing

```python
test_adata, de_results = diff_exp_ontology(
    adata=adata_external,
    ontology_keys="ontology_scores",
    groupby="Status",
    method="wilcoxon",
    pseudo_bulk=False,
)
```

### Pseudobulk testing

```python
test_adata_pb, de_results_pb = diff_exp_ontology(
    adata=adata_external,
    ontology_keys="ontology_scores",
    groupby="Status",
    method="wilcoxon",
    pseudo_bulk=True,
    pseudo_bulk_by=["Patient_Study"],
    summary_metric="mean",
)
```

## Signature enrichment

Allows for querying factors, ligand/receptor signatures, regulons, etc. for enrichment of a user-defined gene list or vector (i.e. external differential expression signature).

### Unweighted curated gene set

```python
sig_results = signature_enrichment(
    signature=["CEBPB", "LGALS3", "JUNB"],
    ontology=ontology,
    search_in=["weights", "regulons", "liana_ligand"],
    cell_types=["Tumor", "Mo_TAM"],
)
```

### Weighted genome-wide signature

```python
weighted_sig = pd.Series(
    [2.1, 1.4, -0.8, 0.6],
    index=["CEBPB", "LGALS3", "MKI67", "ANXA1"],
    name="external_signature",
)

sig_results = signature_enrichment(
    signature=weighted_sig,
    ontology=ontology,
    search_in=["weights", "regulons", "liana_receptor"],
    cell_types="Tumor",
)
```

Each query target returns:

- `results`: ranked enrichment table (GSEA-based for lists, permutation-based for vectors)
- `overlap`: overlap summary between the query and the queried ontology target

Example:

```python
sig_results["weights"]["results"].head()
sig_results["weights"]["overlap"]
```

## Export to Excel

```python
paths = export_ontology_excel(
    ontology=ontology,
    outdir=".",
    prefix="gbm_ontology",
)
print(paths)
```

This writes:

- `gbm_ontology_scores.xlsx`
- `gbm_ontology_features.xlsx`

## Load ontology from Excel

```python
ontology2 = load_ontology_excel(
    scores_xlsx="gbm_ontology_scores.xlsx",
    features_xlsx="gbm_ontology_features.xlsx",
)
```

## End-to-end example

```python
ontology = make_ontology(
    factor_loadings=factor_loadings.fillna(0),
    regulon_loadings=regulon_loadings.fillna(0),
    liana=liana,
    hallmark_lib=hallmark_lib_dict,
    gene_sets=gene_sets_dict,
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

_, de_results = diff_exp_ontology(
    adata=adata_external,
    ontology_keys="ontology_scores",
    groupby="Status",
    method="wilcoxon",
    pseudo_bulk=True,
    pseudo_bulk_by=["Patient_Study"],
    summary_metric="mean",
)

sig_results = signature_enrichment(
    signature=["CEBPB", "LGALS3", "JUNB"],
    ontology=ontology,
    search_in=["weights", "regulons", "liana_ligand"],
    cell_types=["Tumor", "Mo_TAM"],
)

paths = export_ontology_excel(
    ontology=ontology,
    outdir=".",
    prefix="gbm_ontology",
)
```


