add_op_to_compiled_list()
# The upstream binary packager derives one kernel name from each directory.
# Expose the two extra entry points as generated, source-identical staging
# directories. Canonical sources stay in this operator; no toolkit edits.
get_filename_component(tree_gdn_root ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
string(SHA256 tree_gdn_build_id "${CMAKE_BINARY_DIR}")
foreach(compact_name tree_gdn_compact_verify_v310 tree_gdn_compact_replay_v310)
    # add_ops_src_copy filters any absolute path containing 'op_host'.
    # Use a per-build staging directory outside that name, still in this op.
    set(compact_stage ${tree_gdn_root}/.compact_build_sources/${tree_gdn_build_id}/${compact_name})
    file(MAKE_DIRECTORY ${compact_stage}/op_kernel)
    foreach(compact_file ${compact_name}.cpp compact_layout.h compact_verify.h compact_replay.h tree_gated_delta_rule_v310_tiling_data.h)
        configure_file(${tree_gdn_root}/op_kernel/${compact_file}
            ${compact_stage}/op_kernel/${compact_file} COPYONLY)
    endforeach()
    set(COMPILED_OPS ${COMPILED_OPS} ${compact_name} CACHE STRING "Compiled Ops" FORCE)
    set(COMPILED_OP_DIRS ${COMPILED_OP_DIRS} ${compact_stage} CACHE STRING "Compiled Ops Dirs" FORCE)
endforeach()
# The upstream stub otherwise writes into PROJECT_SOURCE_DIR/build even for
# an out-of-tree build. Keep this build artifact with the selected binary dir.
foreach(stub_target optiling opapi_math)
    if (TARGET ${stub_target})
        set_target_properties(${stub_target} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
    endif()
endforeach()
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE tree_gated_delta_rule_v310_def.cpp
        tree_gdn_compact_verify_v310_def.cpp tree_gdn_compact_replay_v310_def.cpp)
endif()
set(tree_gdn_compile_options --cce-auto-sync=on -Wno-deprecated-declarations)
option(TREE_GDN_MANUAL_SYNC "Use the validated explicit tree-GDN pipeline fences" ON)
if (TREE_GDN_MANUAL_SYNC OR "$ENV{TREE_GDN_MANUAL_SYNC}" STREQUAL "1")
    set(tree_gdn_compile_options --cce-auto-sync=off -Wno-deprecated-declarations)
endif()
option(TREE_GDN_SANITIZE "Build tree GDN with CANN sanitizer instrumentation" OFF)
if (TREE_GDN_SANITIZE OR "$ENV{TREE_GDN_SANITIZE}" STREQUAL "1")
    # TBE -sanitizer enables BOTH compiler instrumentation and runtime linkage.
    # Passing --cce-enable-sanitizer alone leaves unresolved report symbols.
    list(APPEND tree_gdn_compile_options -sanitizer)
endif()
add_ops_compile_options(
    OP_NAME TreeGatedDeltaRuleV310
    OPTIONS ${tree_gdn_compile_options}
)
if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE tree_gated_delta_rule_v310 tree_gdn_compact_verify_v310 tree_gdn_compact_replay_v310
        ACLNNTYPE aclnn_exclude aclnn_exclude aclnn_exclude)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()
foreach(compact_op TreeGdnCompactVerifyV310 TreeGdnCompactReplayV310)
    add_ops_compile_options(OP_NAME ${compact_op} OPTIONS ${tree_gdn_compile_options})
endforeach()
