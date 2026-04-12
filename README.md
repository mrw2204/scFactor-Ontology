
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

## Install
```bash
pip install "scfo @ git+https://github.com/mrw2204/scFactor-Ontology.git@main"
```

## How to use
See `docs/vignette.md` for a full end-to-end example and ReadTheDocs for detailed tutorials: https://scfactor-ontology.readthedocs.io/en/latest/

## Example data (GBM and low-grade glioma ontologies):
Google Drive with example data: https://drive.google.com/drive/folders/1V0PoMmk5PseL3cSd1xrHr_btArUtjnhI?usp=sharing

**To request access to example data on Google Drive, email Matt Warren at mrw2204@cumc.columbia.edu**

## Licensing
*Licensing for this repository is currently under discussion with collaborators/institution. Until then, all rights are reserved unless explicit permission is granted.*
