#!/bin/bash
set -euo pipefail

JOB_ROOT="。/"

submitted_count=0
shopt -s nullglob
for job_dir in "$JOB_ROOT"/*; do
  if [[ -d "$job_dir" && -f "$job_dir/newvasp.sh" ]]; then
    (
      cd "$job_dir"
      echo "Submitting: $job_dir"
      sbatch newvasp.sh
    )q
    submitted_count=$((submitted_count + 1))
  fi
done

echo "submitted_count=$submitted_count"
