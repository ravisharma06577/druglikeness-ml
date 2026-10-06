Extended analyses (steps 6-10) for the drug-likeness paper
===========================================================
06_extract_extended.py     pulls approved drugs, phase 2-3 compounds (never approved) and a 20,000-compound
                           random pool of other small molecules from ChEMBL release 37, with first-report year
07_prepare_pool.py         curates the pool and computes all feature sets (10 descriptors, all RDKit descriptors,
                           Morgan bits and counts, MACCS keys)
08_negative_sensitivity.py repeats the main comparison with four definitions of the "other" class
                           (random, property-matched, era-matched, phase 2-3 never approved), 5 seeds each
09_baselines_time_era.py   QED, Veber, Ghose, rule of five, MW+logP, year-only and nearest-neighbour baselines;
                           time-based split; era check; precision at realistic prevalence; calibration
10_features_cv_pdp.py      wider feature panels, repeated grouped cross-validation, partial dependence
dl_common.py               shared helper functions (do not run directly)
run_extended.sh            runs 06-10 in order and saves the printout to Results/extended_output.txt

Fixed hyperparameters (from the primary tuned analysis) are used throughout steps 8-10, so no tuning happens
on any test set. Optional speed settings (environment variables): DL_N_SEEDS (default 5), DL_N_REPEATS (default 5),
DL_N_OTHER (default 20000).
