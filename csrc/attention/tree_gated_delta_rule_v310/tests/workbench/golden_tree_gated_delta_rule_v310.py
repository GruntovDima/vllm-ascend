"""P6 deterministic inputs + native one-token NPU golden, FP16 at every edge.

Adapted from gen_data.py.tmpl. The primary oracle is the existing native
310P recurrence, not a NumPy reimplementation claiming device correctness.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def comb(width, depth):
    parents, primary = [-1], 0
    for _ in range(depth):
        next_primary = len(parents)
        parents.extend([primary] * width)
        primary = next_primary
    return parents


def cases():
    return [
        ("root_h1_r16", [-1], 1, 1, 16),
        ("chain_h3_r16", comb(1, 4), 1, 3, 16),
        ("siblings_h8_r32", comb(16, 1), 4, 8, 32),
        ("comb2d4_h8_r32", comb(2, 4), 4, 8, 32),
        ("comb8d2_h32_r32", comb(8, 2), 16, 32, 32),
        ("comb16d4_h32_r32", comb(16, 4), 16, 32, 32),
        ("mixed_h8_r16", [-1, 0, 0, 1, 2, 1, 3, 4, 5, 6, 7, 8], 2, 8, 16),
        ("wide_h32_r64", comb(16, 1), 16, 32, 64),
        ("chain_h32_r64", comb(1, 4), 16, 32, 64),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--case", default="all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    selected = [c for c in cases() if args.case in ("all", c[0])]
    if not selected:
        parser.error("Unknown case")
    root = Path(__file__).resolve().parents[2]
    out_dir = args.out.resolve()
    if not out_dir.is_relative_to(root):
        parser.error("Output must stay under this operator workspace")
    if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        parser.error("Select the authorized physical NPU before starting")

    import torch
    import torch_npu
    from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op

    bootstrap_custom_op_env()
    torch_npu.npu.set_device(args.device)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    device = f"npu:{args.device}"
    torch.manual_seed(args.seed)
    for case_id, parents, hk, hv, tile in selected:
        directory = out_dir / case_id
        if directory.exists():
            raise FileExistsError(f"Refusing to overwrite frozen inputs: {directory}")
        directory.mkdir(parents=True)
        n = len(parents)
        rng = np.random.RandomState(args.seed)
        def normal(shape):
            return torch.from_numpy(rng.normal(size=shape).astype(np.float32))
        cpu = {
            "query": torch.nn.functional.normalize(normal((n, hk, 128)), dim=-1).half(),
            "key": torch.nn.functional.normalize(normal((n, hk, 128)), dim=-1).half(),
            "value": normal((n, hv, 128)).half(),
            "beta": torch.from_numpy(rng.uniform(size=(n, hv)).astype(np.float16)),
            "g": torch.from_numpy(-rng.uniform(size=(n, hv)).astype(np.float32)),
            "initial_state": (normal((hv, 128, 128)) * 0.1).half(),
        }
        for name, tensor in cpu.items():
            tensor.numpy().tofile(directory / f"input_{name}.bin")
        tensors = {name: tensor.to(device) for name, tensor in cpu.items()}
        cu = torch.tensor([1], dtype=torch.int32, device=device)
        indices = torch.tensor([0], dtype=torch.int32, device=device)
        snapshots, outputs = [], []
        with torch.inference_mode():
            for node, parent in enumerate(parents):
                state = (tensors["initial_state"] if parent < 0 else snapshots[parent]).unsqueeze(0).clone()
                out = torch.ops._C_ascend.npu_recurrent_gated_delta_rule_310(
                    query=tensors["query"][node:node+1].contiguous(),
                    key=tensors["key"][node:node+1].contiguous(),
                    value=tensors["value"][node:node+1].contiguous(),
                    beta=tensors["beta"][node:node+1].contiguous(),
                    state=state, actual_seq_lengths=cu, ssm_state_indices=indices,
                    g=tensors["g"][node:node+1].contiguous(), gk=None,
                    num_accepted_tokens=None, scale_value=128 ** -0.5)
                snapshots.append(state[0].clone())
                outputs.append(out.clone())
        torch_npu.npu.synchronize()
        golden = {"out": torch.cat(outputs).cpu().numpy(),
                  "snapshots": torch.stack(snapshots).cpu().numpy()}
        for name, array in golden.items():
            if not np.isfinite(array).all():
                raise RuntimeError(f"Nonfinite native golden: {case_id}/{name}")
            array.tofile(directory / f"golden_{name}.bin")
        params = [str(n), str(hk), str(hv), str(tile), repr(128 ** -0.5)]
        (directory / "params.txt").write_text(" ".join(params) + "\n" + " ".join(map(str, parents)) + "\n")
        manifest = {"case": case_id, "parents": parents, "n": n, "hk": hk, "hv": hv,
                    "v_tile": tile, "seed": args.seed, "golden_source": "existing_native_one_token_npu",
                    "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in directory.glob("*.bin")},
                    "torch": torch.__version__, "torch_npu": torch_npu.__version__}
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print("GOLDEN " + case_id, flush=True)


if __name__ == "__main__":
    main()
