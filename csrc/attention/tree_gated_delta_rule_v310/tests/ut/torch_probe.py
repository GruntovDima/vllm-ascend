"""Isolated C4 adapter probe: build, Meta, native functional/out, strict oracle.

No vLLM import or installed-package mutation. Output dumps and report stay in
this operator's workspace. Inputs and goldens are immutable native artifacts.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    import numpy as np
    import torch
    import torch_npu
    from torch.utils.cpp_extension import load

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--case", default="all")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    build = ROOT / "op_host/_torch_probe"
    build.mkdir(exist_ok=True)
    cann = Path(os.environ["ASCEND_HOME_PATH"])
    npu = Path(torch_npu.__file__).resolve().parent
    bridge = ROOT.parents[1] / "aclnn_torch_adapter"
    sources = [ROOT / "tests/ut/torch_binding_probe.cpp",
               bridge / "NPUBridge.cpp", bridge / "NPUStorageImpl.cpp"]
    load(name="tree_gdn_binding_probe", sources=[str(x) for x in sources],
         extra_include_paths=[str(npu.parent), str(npu / "include"), str(cann / "include")],
         extra_cflags=["-O2", "-std=c++17"],
         extra_ldflags=[f"-L{npu / 'lib'}", "-ltorch_npu",
                        f"-L{cann / 'lib64'}", "-lascendcl"],
         build_directory=str(build), is_python_module=False, verbose=True)

    @torch.library.register_fake("tree_gdn_probe::forward")
    def fake(query, key, value, beta, initial, g, parents, scale, v_tile=0):
        return value.new_empty(value.shape), initial.new_empty((value.shape[0], *initial.shape))

    @torch.library.register_fake("tree_gdn_probe::forward.out")
    def fake_out(query, key, value, beta, initial, g, parents, scale, v_tile=0, *, out, snapshots):
        return out, snapshots

    q = torch.empty((9, 4, 128), dtype=torch.float16, device="meta")
    v = torch.empty((9, 8, 128), dtype=torch.float16, device="meta")
    b = torch.empty((9, 8), dtype=torch.float16, device="meta")
    initial = torch.empty((8, 128, 128), dtype=torch.float16, device="meta")
    g = torch.empty((9, 8), dtype=torch.float32, device="meta")
    fake_result = torch.ops.tree_gdn_probe.forward(q, q, v, b, initial, g, [-1, 0, 0, 1, 1, 3, 3, 5, 5], 128**-0.5)
    assert [x.shape for x in fake_result] == [v.shape, (9, 8, 128, 128)]
    assert all(x.dtype == torch.float16 and x.device.type == "meta" for x in fake_result)
    print("PASS: compiled adapter + Meta schema", flush=True)
    if args.build_only:
        return
    assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), "Select the authorized physical NPU"
    torch_npu.npu.set_device(args.device)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    device = f"npu:{args.device}"
    spec = importlib.util.spec_from_file_location("tree_compare", ROOT / "tests/workbench/compare_outputs.py")
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    report = []
    cases = (ROOT / "tests/workbench/shapes_light.txt").read_text().split()
    for case_id in cases:
        if args.case not in ("all", case_id):
            continue
        directory = ROOT / "tests/workbench/data_native_v1" / case_id
        manifest = json.loads((directory / "manifest.json").read_text())
        for filename, digest in manifest["sha256"].items():
            assert hashlib.sha256((directory / filename).read_bytes()).hexdigest() == digest
        lines = (directory / "params.txt").read_text().splitlines()
        n, hk, hv, tile, scale = lines[0].split()
        n, hk, hv, tile = map(int, (n, hk, hv, tile))
        parents = list(map(int, lines[1].split()))
        shapes = [(n, hk, 128), (n, hk, 128), (n, hv, 128), (n, hv),
                  (hv, 128, 128), (n, hv)]
        names = ["query", "key", "value", "beta", "initial_state", "g"]
        cpu = [np.fromfile(directory / f"input_{name}.bin",
                          dtype=np.float32 if name == "g" else np.float16).reshape(shape)
               for name, shape in zip(names, shapes)]
        inputs = [torch.from_numpy(array.copy()).to(device) for array in cpu]
        expected = [np.fromfile(directory / f"golden_{name}.bin", np.float16).reshape(shape)
                    for name, shape in [("out", (n, hv, 128)), ("snapshots", (n, hv, 128, 128))]]
        results = []
        with torch.inference_mode():
            for kind in ("functional", "out"):
                for repeat in range(3):
                    if kind == "functional":
                        output = torch.ops.tree_gdn_probe.forward(*inputs, parents, float(scale), tile)
                    else:
                        target = [torch.empty(x.shape, device=device, dtype=torch.float16) for x in expected]
                        output = torch.ops.tree_gdn_probe.forward.out(
                            *inputs, parents, float(scale), tile, out=target[0], snapshots=target[1])
                        assert all(a.data_ptr() == b.data_ptr() for a, b in zip(target, output))
                    torch_npu.npu.synchronize()
                    row = dict(kind=kind, repeat=repeat, metrics=[])
                    for name, tensor, golden in zip(("out", "snapshots"), output, expected):
                        observed = tensor.cpu().numpy()
                        observed.tofile(directory / f"output_torch_{kind}_{name}_r{repeat}.bin")
                        metrics = [compare.metrics(observed[i], golden[i]) for i in range(n)]
                        assert all(x["passed"] for x in metrics), (case_id, kind, repeat, name, metrics)
                        row["metrics"].append(dict(name=name, min_cosine=min(x["cosine"] for x in metrics), exact=True))
                    results.append(row)
                assert all(np.array_equal(t.cpu().numpy(), original) for t, original in zip(inputs, cpu)), "input mutated"
        # Failure paths must reject before launch without poisoning the next call.
        negatives = 0
        with torch.inference_mode():
            target = [torch.empty(x.shape, device=device, dtype=torch.float16) for x in expected]
            for label, bad_parents, bad_out in (("invalid_parent", [0] + parents[1:], target[0]), ("output_alias", parents, inputs[2])):
                try:
                    torch.ops.tree_gdn_probe.forward.out(
                        *inputs, bad_parents, float(scale), tile, out=bad_out, snapshots=target[1])
                except RuntimeError:
                    negatives += 1
                    print(f"PASS: negative {label}", flush=True)
                else:
                    raise AssertionError(f"Invalid call accepted: {label}")
            assert negatives == 2
            torch.ops.tree_gdn_probe.forward.out(
                *inputs, parents, float(scale), tile, out=target[0], snapshots=target[1])
            torch_npu.npu.synchronize()
            assert all(np.array_equal(t.cpu().numpy(), gold) for t, gold in zip(target, expected))

        graph_rows = []
        if args.graph:
            with torch.inference_mode():
                static_inputs = [t.clone() for t in inputs]
                graph_out = [torch.empty(x.shape, device=device, dtype=torch.float16) for x in expected]
                stream = torch_npu.npu.Stream()
                stream.wait_stream(torch_npu.npu.current_stream())
                with torch_npu.npu.stream(stream):
                    for _ in range(3):
                        torch.ops.tree_gdn_probe.forward.out(
                            *static_inputs, parents, float(scale), tile, out=graph_out[0], snapshots=graph_out[1])
                stream.synchronize()
                graph = torch_npu.npu.NPUGraph()
                with torch_npu.npu.graph(graph, stream=stream):
                    torch.ops.tree_gdn_probe.forward.out(
                        *static_inputs, parents, float(scale), tile, out=graph_out[0], snapshots=graph_out[1])
                original_graph = None
                for factor in (1.0, 0.5, -0.75):
                    for dst, src in zip(static_inputs, inputs):
                        dst.copy_(src)
                    static_inputs[0].mul_(factor)
                    static_inputs[4].mul_(factor)
                    eager = torch.ops.tree_gdn_probe.forward(*static_inputs, parents, float(scale), tile)
                    torch_npu.npu.synchronize()
                    reference = [t.cpu().numpy().copy() for t in eager]
                    graph.replay()
                    torch_npu.npu.synchronize()
                    observed = [t.cpu().numpy().copy() for t in graph_out]
                    assert all(np.array_equal(a, b) for a, b in zip(observed, reference)), "graph differs from eager"
                    if original_graph is None:
                        original_graph = observed
                        assert all(np.array_equal(a, b) for a, b in zip(observed, expected))
                    else:
                        assert any(not np.array_equal(a, b) for a, b in zip(observed, original_graph)), "graph froze tensor values"
                    graph_rows.append(dict(input_factor=factor, exact_vs_eager=True))
                graph.reset()
            print(f"PASS: {case_id} graph replay with 3 changing inputs", flush=True)
        report.append(dict(case=case_id, passed=True, repeats=results,
                           negative_checks=negatives, graph_replays=graph_rows))
        print(f"PASS: {case_id} functional/out x3, every node bit-exact", flush=True)
    assert report, "No cases selected"
    suffix = "_graph" if args.graph else ""
    (ROOT / f"workbench/analysis/torch_probe{suffix}_report.json").write_text(json.dumps({
        "evidence_type": "device", "source_sha256": hashlib.sha256(
            (ROOT / "op_kernel/tree_gated_delta_rule_v310.h").read_bytes()).hexdigest(),
        "results": report, "passed": True}, indent=2) + "\n")


if __name__ == "__main__":
    main()
