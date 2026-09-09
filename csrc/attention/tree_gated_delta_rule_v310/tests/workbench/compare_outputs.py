"""Compare real device dumps with the native per-node golden (P6 template).

Strict gate: bit-exact outputs and checkpoints plus cosine >= 0.999999.
All repeats must match. Reports per-node minima so aggregation cannot hide
a corrupted branch. Finite values and exact file sizes are mandatory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

THRESHOLD = 0.999999


def metrics(observed, golden):
    if observed.shape != golden.shape or not (np.isfinite(observed).all() and np.isfinite(golden).all()):
        return {"passed": False, "reason": "shape mismatch or nonfinite data"}
    a, b = observed.astype(np.float64).reshape(-1), golden.astype(np.float64).reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    cosine = float(np.dot(a, b) / denom) if denom else float(np.array_equal(a, b))
    error = np.abs(a - b)
    return {"passed": bool(np.array_equal(observed, golden) and cosine >= THRESHOLD),
            "cosine": cosine, "exact": bool(np.array_equal(observed, golden)),
            "max_abs_error": float(error.max()), "different_elements": int(np.count_nonzero(error))}


def compare(directory, repeats):
    manifest = json.loads((directory / "manifest.json").read_text())
    for filename, digest in manifest["sha256"].items():
        if hashlib.sha256((directory / filename).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Frozen data changed: {filename}")
    shapes = {"out": (manifest["n"], manifest["hv"], 128),
              "snapshots": (manifest["n"], manifest["hv"], 128, 128)}
    report = {"evidence_type": "device", "case": manifest["case"], "repeats": [],
              "passed": True, "cosine": 1.0, "golden_source": manifest["golden_source"]}
    first = {}
    for repeat in range(repeats):
        row = {}
        for name, shape in shapes.items():
            golden = np.fromfile(directory / f"golden_{name}.bin", dtype=np.float16).reshape(shape)
            observed = np.fromfile(directory / f"output_{name}_r{repeat}.bin", dtype=np.float16).reshape(shape)
            item = metrics(observed, golden)
            item["nodes"] = [metrics(observed[i], golden[i]) for i in range(shape[0])]
            item["repeat_exact"] = name not in first or np.array_equal(first[name], observed)
            item["passed"] &= item["repeat_exact"] and all(n["passed"] for n in item["nodes"])
            first.setdefault(name, observed)
            report["passed"] &= bool(item["passed"])
            report["cosine"] = min(report["cosine"], *(n.get("cosine", 0.0) for n in item["nodes"]))
            row[name] = item
        report["repeats"].append(row)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("At least three repeats required")
    report = compare(args.case_dir, args.repeats)
    (args.case_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"cosine: {report['cosine']:.12f}")
    print("PASS" if report["passed"] else "FAIL")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
