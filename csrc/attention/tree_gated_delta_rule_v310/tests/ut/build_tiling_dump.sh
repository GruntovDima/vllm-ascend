#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
cann="${ASCEND_HOME_PATH:?Source CANN first}"
asc="$cann/aarch64-linux/asc"
g++ -std=c++17 -O2 -D_GLIBCXX_USE_CXX11_ABI=0 "$root/tests/ut/dump_tree_tiling.cpp" \
 -I"$cann/include" -I"$cann/include/experiment" -I"$asc/include" -I"$asc/include/tiling" \
 -L"$cann/lib64" -ltiling_api -lplatform -lgraph -lgraph_base -lregister -lascendalog -lunified_dlog \
 -o "$root/tests/ut/dump_tree_tiling"
