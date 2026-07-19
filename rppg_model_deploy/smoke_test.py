import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rppg_model_deploy.adapter import RppgModelAdapter
    from rppg_model_deploy.registry import (
        DEFAULT_MODEL_KEY,
        DEPLOYED_MODEL_KEYS,
        EXTERNAL_CANDIDATES,
        check_assets,
        get_model_spec,
        model_specs_as_dict,
    )
else:
    from .adapter import RppgModelAdapter
    from .registry import (
        DEFAULT_MODEL_KEY,
        DEPLOYED_MODEL_KEYS,
        EXTERNAL_CANDIDATES,
        check_assets,
        get_model_spec,
        model_specs_as_dict,
    )


def make_synthetic_faces(shape, bpm=72.0, fps=30.0):
    frames, height, width, _ = shape
    t = np.arange(frames, dtype=np.float32) / fps
    pulse = np.sin(2 * math.pi * (bpm / 60.0) * t)

    yy = np.linspace(0, 1, height, dtype=np.float32)[None, :, None]
    xx = np.linspace(0, 1, width, dtype=np.float32)[None, None, :]
    base = np.zeros((frames, height, width, 3), dtype=np.float32)
    base[..., 0] = 118 + 4 * yy
    base[..., 1] = 128 + 7 * pulse[:, None, None] + 3 * xx
    base[..., 2] = 102 + 2 * yy
    return np.clip(base, 0, 255).astype(np.uint8)


def parse_model_keys(raw):
    if raw == "all":
        return list(DEPLOYED_MODEL_KEYS)
    return [item.strip() for item in raw.split(",") if item.strip()]


def smoke_one(model_key):
    spec = get_model_spec(model_key)
    assets = check_assets(spec.key)
    faces = make_synthetic_faces(spec.input_shape)
    adapter = RppgModelAdapter(spec.key)

    load_start = time.perf_counter()
    adapter.load()
    load_s = time.perf_counter() - load_start

    infer_start = time.perf_counter()
    bvp = adapter.direct_bvp(faces)
    infer_s = time.perf_counter() - infer_start

    finite = bool(np.isfinite(bvp).all())
    return {
        "model": spec.key,
        "display_name": spec.display_name,
        "backend_model": spec.model_name,
        "status": "ok" if finite and bvp.size else "failed",
        "asset_check": assets,
        "input_shape": list(spec.input_shape),
        "bvp_shape": list(bvp.shape),
        "bvp_mean": float(np.mean(bvp)) if bvp.size else None,
        "bvp_std": float(np.std(bvp)) if bvp.size else None,
        "finite": finite,
        "load_seconds": round(load_s, 3),
        "inference_seconds": round(infer_s, 3),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Smoke test deployed lightweight rPPG models.")
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model keys, model names, or 'all'.",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "smoke_results.json"),
        help="Path to write the JSON report.",
    )
    args = parser.parse_args(argv)

    selected = parse_model_keys(args.models)
    results = []
    for key in selected:
        try:
            results.append(smoke_one(key))
        except Exception as exc:
            try:
                spec = get_model_spec(key)
                model = spec.key
                display_name = spec.display_name
            except Exception:
                model = key
                display_name = key
            results.append(
                {
                    "model": model,
                    "display_name": display_name,
                    "status": "failed",
                    "error": repr(exc),
                }
            )

    ok_count = sum(item.get("status") == "ok" for item in results)
    payload = {
        "default_model": DEFAULT_MODEL_KEY,
        "summary": {
            "selected": len(results),
            "ok": ok_count,
            "failed": len(results) - ok_count,
        },
        "deployed_models": model_specs_as_dict(DEPLOYED_MODEL_KEYS),
        "external_candidates": EXTERNAL_CANDIDATES,
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print("wrote", output_path)
    return 0 if ok_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
