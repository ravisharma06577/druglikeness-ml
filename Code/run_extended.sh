#!/bin/bash
# Runs the extended analyses (steps 6-10) one after another and saves everything printed to
# ../Results/extended_output.txt. Use:  caffeinate -i bash run_extended.sh
set -o pipefail
cd "$(dirname "$0")"
mkdir -p ../Results
for s in 06_extract_extended.py 07_prepare_pool.py 08_negative_sensitivity.py 09_baselines_time_era.py 10_features_cv_pdp.py; do
  echo "=============== $s  ($(date)) ===============" | tee -a ../Results/extended_output.txt
  python "$s" 2>&1 | tee -a ../Results/extended_output.txt
  if [ ${PIPESTATUS[0]} -ne 0 ]; then echo "STOPPED: $s failed. Send me the lines above."; exit 1; fi
done
echo "ALL DONE ($(date))" | tee -a ../Results/extended_output.txt
