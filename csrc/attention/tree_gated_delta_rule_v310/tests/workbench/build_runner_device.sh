#!/usr/bin/env bash
# P6 device runner build: toolkit/build paths supplied by caller, no baked-in coords.
set -euo pipefail
OPERATOR_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
: "${ASCEND_HOME_PATH:?Source the selected CANN set_env.sh first}"
TREE_BUILD_DIR="${TREE_BUILD_DIR:-$OPERATOR_ROOT/op_host/_build}"
c++ -std=c++17 -O2 -D_GLIBCXX_USE_CXX11_ABI=0 \
  "$OPERATOR_ROOT/tests/workbench/run_tree_gated_delta_rule_v310_dump.cpp" \
  -I"$ASCEND_HOME_PATH/include" -I"$ASCEND_HOME_PATH/aarch64-linux/include" \
  -L"$TREE_BUILD_DIR" -L"$ASCEND_HOME_PATH/lib64" \
  -Wl,-rpath,"$TREE_BUILD_DIR" -lcust_opapi -lascendcl -lnnopbase \
  -o "$OPERATOR_ROOT/tests/workbench/run_tree_gated_delta_rule_v310_dump"
