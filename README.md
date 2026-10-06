# Drug-likeness machine-learning analysis (ChEMBL release 37)

Code, curated data lists, result tables and figures for the manuscript

> **Machine-Learning Classification of Approved Drugs Versus Other Small Molecules: How Performance Depends on the Negative-Class Definition, Compound Era and Evaluation Design**
> Ravi Sharma, Department of Pharmaceutical Sciences, Maharshi Dayanand University, Rohtak, Haryana, India



## What this project does

The project asks how far the apparent performance of machine-learning classifiers of approved small-molecule drugs depends on how the "other" (negative) class is defined and how the data are split. Using ChEMBL release 37 it:

- compares logistic regression, support vector machine, random forest and gradient boosting on ten physicochemical descriptors, Morgan fingerprints, or both;
- repeats the comparison with four negative classes (random, property-matched, era-matched, and phase 2-3 compounds never approved), five draws each;
- compares the models with the rule of five, Veber and Ghose filters, QED, and simple property, era and similarity baselines;
- uses scaffold-disjoint, time-based and repeated grouped validation, calibration, expected precision at realistic prevalence, wider feature panels and partial dependence;
- examines feature importance, the applicability domain and the errors of the best model.

Headline results (random forest, descriptors plus fingerprints, mean test ROC-AUC over five draws): 0.831 against random other compounds, 0.765 against property-matched and against era-matched compounds, and 0.591 against phase 2-3 compounds that were never approved. QED reached an AUC of 0.507. See the manuscript and the files in `Results/` for all numbers.

## Repository layout

```
Code/        analysis scripts (01-10), shared helpers, run script
Clean_Data/  curated compound lists and calculated descriptors (large .npy feature arrays are not included)
Results/     result tables (CSV), logs and printed output (saved models are not included)
Figures/     all figures as PNG
README.md    this file
LICENSE      MIT licence for the code
```

Not included in the repository (to keep it small, and because they can be regenerated): the raw ChEMBL database and `Original_Data/`, the saved model files in `Results/models/`, and the large feature arrays `Clean_Data/*.npy`.

## Scripts

| Script | Purpose |
|---|---|
| `01_extract.py` | Extract approved small molecules and a random sample of other small molecules from ChEMBL |
| `02_prepare.py` | Curate structures, balance classes, calculate ten descriptors and Morgan fingerprints, first plots |
| `03_models.py` | Scaffold-disjoint split, hyperparameter tuning, test evaluation, rule-of-five baseline, ROC curves |
| `04_analysis.py` | Bootstrap confidence intervals, permutation importance, fingerprint bits, applicability domain, error analysis |
| `05_robustness.py` | Five additional scaffold splits and a label-shuffling control |
| `06_extract_extended.py` | Extract approved, phase 2-3 and a 20,000-compound pool of other compounds, with first-report year |
| `07_prepare_pool.py` | Curate the pool; calculate all 217 RDKit descriptors, Morgan bits and counts, MACCS keys |
| `08_negative_sensitivity.py` | Four negative-class definitions, five draws each |
| `09_baselines_time_era.py` | Baselines, time-based split, era check, expected precision at realistic prevalence, calibration |
| `10_features_cv_pdp.py` | Feature panels, repeated grouped cross-validation, partial dependence |
| `dl_common.py` | Shared helper functions for scripts 06-10 |
| `run_extended.sh` | Runs scripts 06-10 in order and saves the printout |

## Requirements

Tested on an Apple M4 MacBook with macOS and conda (Miniforge):

- Python 3.11.16
- RDKit 2026.03.6
- scikit-learn 1.9.1
- NumPy 2.4.6
- pandas 3.0.6
- matplotlib 3.11.2
- joblib 1.6.0
- chembl-downloader (to download the ChEMBL database)

Create the environment:

```bash
mamba create -n druglike python=3.11 -y
mamba activate druglike
mamba install -c conda-forge rdkit -y
pip install pandas numpy scikit-learn matplotlib joblib jupyterlab chembl-downloader
```

(`conda` can be used instead of `mamba`.)

## How to reproduce

**1. Get the data.** Download ChEMBL release 37 in SQLite format:

```bash
python -c "import chembl_downloader; print(chembl_downloader.latest()); print(chembl_downloader.download_extract_sqlite())"
```

This needs tens of gigabytes of free disk space once the database is unpacked. The scripts expect the database at `~/.data/chembl/37/chembl_37.db`.

**2. Folder layout.** Scripts 01-05 assume the project folder is `~/Drug_Likeness_ML` with the subfolders `Original_Data`, `Clean_Data`, `Code`, `Results` and `Figures`. Create them and copy this repository's content into place:

```bash
mkdir -p ~/Drug_Likeness_ML/{Original_Data,Clean_Data,Code,Results,Figures}
```

Scripts 06-10 read two optional environment variables instead: `DL_BASE` (project folder, default `~/Drug_Likeness_ML`) and `DL_DB` (path to the ChEMBL database).

**3. Primary analysis** (run from the `Code` folder):

```bash
cd ~/Drug_Likeness_ML/Code
python 01_extract.py
python 02_prepare.py
python 03_models.py
python 04_analysis.py
python 05_robustness.py
```

**4. Extended analyses** (about 20 minutes on an Apple M4):

```bash
caffeinate -i bash run_extended.sh     # macOS; on other systems: bash run_extended.sh
```

Optional settings for faster test runs: `DL_N_SEEDS` (default 5), `DL_N_REPEATS` (default 5), `DL_N_OTHER` (default 20000).

Random seeds are fixed in every script (42 for the main analyses, 1-5 for repeated splits and draws), so results should reproduce on the same software versions. Small numerical differences may appear with other versions or operating systems.

## Method notes

- **Labels.** Approved = ChEMBL maximum phase 4. Other = no recorded clinical phase. Phase 2-3 = maximum phase 2 or 3 (never recorded as approved). "Other" does not mean "proven non-drug".
- **Curation.** Largest fragment, neutralisation, elements H C N O F P S Cl Br I, 10-70 heavy atoms, duplicates removed without stereochemistry.
- **Era.** First-report year = year of the earliest ChEMBL document in which a compound is recorded.
- **Splits.** Bemis-Murcko scaffold-disjoint splits; a time-based split (training on compounds first reported up to 2015); repeated grouped cross-validation.
- **Hyperparameters.** Tuned by grouped cross-validation in the primary analysis (script 03); fixed at the selected values for the extended analyses (scripts 08-10).
- **Known detail.** scikit-learn's `predict()` assigns a compound with probability exactly 0.5 to the other class, while the error and applicability-domain analysis in `04_analysis.py` assigns it to the approved class. This affects one test compound (186 versus 187 errors) and is disclosed in the manuscript.

## Data and licences

- **Code:** MIT licence (see `LICENSE`).
- **Source data:** ChEMBL (https://www.ebi.ac.uk/chembl/), released under a Creative Commons Attribution-ShareAlike licence (check the licence version on the ChEMBL website). The compound lists and descriptors in `Clean_Data/` and the tables in `Results/` are derived from ChEMBL and should be used under the same terms. Please cite ChEMBL when using them:
  Zdrazil B, Felix E, Hunter F, et al. (2024) The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods. Nucleic Acids Res 52:D1180-D1192.

## How to cite

Sharma R. Machine-learning classification of approved drugs versus other small molecules: how performance depends on the negative-class definition, compound era and evaluation design. *[Journal and year to be added after publication]*. Code and data archive: *[Zenodo DOI]*.

## Contact

Ravi Sharma, Department of Pharmaceutical Sciences, Maharshi Dayanand University, Rohtak, Haryana, India. Please open an issue in this repository for questions.
