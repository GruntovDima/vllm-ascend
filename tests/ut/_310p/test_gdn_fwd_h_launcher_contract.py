# SPDX-License-Identifier: Apache-2.0
"""Source and host-stub checks for the FwdH launcher/header contract.

These tests need neither vLLM nor CANN. Host-stub compilation checks dispatch
and argument counts only; it does not compile or execute the device kernels.
"""

import re
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OP_KERNEL = ROOT / "csrc/moe/chunk_gated_delta_rule_fwd_h/op_kernel"
LAUNCHER = OP_KERNEL / "chunk_gated_delta_rule_fwd_h.cpp"

# Deliberately no TileShapes, kGated, gk or useGk: the checked-in arch20 and
# arch22 declarations both expose the historical four-parameter interface.
HOST_STUB = """
#define __global__
#define __aicore__
#define __gm__
#define KERNEL_TASK_TYPE_DEFAULT(...)
using GM_ADDR = void*;
using half = float;
using bfloat16_t = double;
struct ChunkGatedDeltaRuleFwdHTilingData {
    int dataType;
    int stateDataType;
    int gDataType;
};
namespace AscendC {
inline GM_ADDR GetUserWorkspace(GM_ADDR workspace) { return workspace; }
}
namespace Catlass::Gemm::Kernel {
template<class Input, class Gate, class State, class Workspace>
struct GDNFwdHKernel {
    void Init(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
              GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR) {}
    void Process() {}
};
}
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200)
#define CATLASS_UNIFIED_CORE 1
#endif
"""


def render_host_stub(source):
    # Quoted includes are checked separately against the real source tree.
    body = re.sub(r'^#include\s+"[^"\n]+"\s*$', "", source, flags=re.MULTILINE)
    return HOST_STUB + body


class FwdHLauncherContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = LAUNCHER.read_text(encoding="utf-8")
        cls.compiler = shutil.which("g++") or shutil.which("clang++")

    def compile_source(self, source, architecture):
        if self.compiler is None:
            self.skipTest("g++ or clang++ required for host-stub syntax checks")
        command = [self.compiler, "-std=c++17", "-fsyntax-only", "-x", "c++"]
        if architecture is not None:
            command.append(f"-D__CCE_AICORE__={architecture}")
        return subprocess.run(
            [*command, "-"],
            input=render_host_stub(source),
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=20,
            check=False,
        )

    def test_real_headers_have_four_template_parameters(self):
        for arch in ("arch20", "arch22"):
            with self.subTest(arch=arch):
                source = (OP_KERNEL / arch / "gemm/kernel/gdn_fwd_h_kernel.hpp").read_text(encoding="utf-8")
                match = re.search(r"template\s*<([^>]+)>\s*class\s+GDNFwdHKernel", source)
                self.assertIsNotNone(match)
                self.assertEqual(len(match.group(1).split(",")), 4)
                init = re.search(r"\bvoid\s+Init\((.*?)\)", source, re.DOTALL)
                self.assertIsNotNone(init)
                self.assertEqual(len(init.group(1).split(",")), 12)
                self.assertNotRegex(init.group(1), r"\bgk\b")

    def test_architecture_includes_exist(self):
        includes = re.findall(r'^#include\s+"(arch[^"\n]+)"', self.source, re.MULTILINE)
        self.assertTrue(includes)
        for include in includes:
            self.assertTrue((OP_KERNEL / include).is_file(), include)

    def test_entry_preserves_twelve_argument_abi(self):
        entries = re.findall(r"\bvoid\s+chunk_gated_delta_rule_fwd_h\((.*?)\)", self.source, re.DOTALL)
        self.assertEqual(len(entries), 1)
        self.assertEqual(len(entries[0].split(",")), 12)
        self.assertNotRegex(entries[0], r"\bgk\b")

    def test_no_dispatch_for_absent_header_contract(self):
        self.assertNotIn("arch35/", self.source)
        self.assertNotIn("GDNFwdHTileShapes", self.source)
        self.assertNotIn("useGk", self.source)

    def test_host_stub_syntax_for_all_existing_architecture_routes(self):
        for architecture in (200, 220, 310, None):
            with self.subTest(architecture=architecture):
                result = self.compile_source(self.source, architecture)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_host_stub_rejects_six_parameter_regression(self):
        mutated = self.source.replace(
            "float, float, workspaceType>", "float, float, workspaceType, int, false>", 1
        )
        self.assertNotEqual(mutated, self.source)
        result = self.compile_source(mutated, 220)
        self.assertNotEqual(result.returncode, 0, "Negative control unexpectedly compiled")
        self.assertRegex(result.stderr, r"wrong number of template arguments|too many template arguments")


if __name__ == "__main__":
    unittest.main()
