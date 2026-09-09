cmake_minimum_required(VERSION 3.20)
project(tree_gdn_icpu LANGUAGES CXX)
enable_testing()
set(tree_cann "$ENV{ASCEND_HOME_PATH}")
if (NOT tree_cann)
    message(FATAL_ERROR "Source the configured CANN environment")
endif()
find_package(tikicpulib CONFIG REQUIRED PATHS "${tree_cann}/tools/cpudebug/lib/cmake" NO_DEFAULT_PATH)
add_executable(test_tree_gdn_icpu test_tree_gdn_icpu.cpp icpu_host_plan.cpp)
target_compile_features(test_tree_gdn_icpu PRIVATE cxx_std_17)
target_compile_options(test_tree_gdn_icpu PRIVATE -O1 -Wno-attributes)
target_include_directories(test_tree_gdn_icpu PRIVATE
 "${tree_cann}/include" "${tree_cann}/include/experiment"
 "${tree_cann}/aarch64-linux/asc/include" "${tree_cann}/aarch64-linux/asc/include/tiling")
target_link_directories(test_tree_gdn_icpu PRIVATE "${tree_cann}/lib64")
target_link_libraries(test_tree_gdn_icpu PRIVATE tikicpulib::Ascend310P3
 tiling_api platform graph graph_base register ascendalog unified_dlog)
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/../../workbench/shapes_light.txt" tree_cases)
foreach(tree_case IN LISTS tree_cases)
    add_test(NAME "tree_gdn_${tree_case}" COMMAND test_tree_gdn_icpu
      "${CMAKE_CURRENT_SOURCE_DIR}/../../workbench/data_native_v1/${tree_case}")
endforeach()
