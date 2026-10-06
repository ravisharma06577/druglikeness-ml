# Drug-likeness machine-learning analysis (ChEMBL release 37)

Code for the manuscript "Machine-Learning Classification of Approved Drugs Versus Other Small Molecules: How Performance Depends on the Negative-Class Definition, Compound Era and Evaluation Design".

Run order (from the Code folder, Python 3.11, RDKit 2026.03.6, scikit-learn 1.9.1):
1. 01_extract.py to 05_robustness.py: primary analysis
2. 06_extract_extended.py to 10_features_cv_pdp.py: extended analyses (or run_extended.sh)

Data: ChEMBL release 37 (https://www.ebi.ac.uk/chembl/), downloaded with chembl-downloader.
Results/ holds the output tables; Figures/ holds the figures. Random seeds are fixed in every script.
