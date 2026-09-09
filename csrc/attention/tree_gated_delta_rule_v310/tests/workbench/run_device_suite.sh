#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
data="${1:?Pass frozen native dataset directory}"
logical="${2:?Pass logical device index}"
: "${ASCEND_RT_VISIBLE_DEVICES:?Select authorized physical NPU}"
while IFS= read -r case_id; do
  [ -n "$case_id" ] || continue
  "$root/tests/workbench/run_tree_gated_delta_rule_v310_dump" "$data/$case_id" "$logical" 3
  python3 "$root/tests/workbench/compare_outputs.py" --case-dir "$data/$case_id" --repeats 3
done < "$root/tests/workbench/shapes_light.txt"
