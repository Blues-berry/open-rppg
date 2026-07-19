# Lightweight rPPG Model Deployment

This folder is a small deployment layer for swapping rPPG models without changing
application code. It uses the bundled open-rppg weights already present in this
repository and exposes a stable adapter.

## Deployed models

| Key | Backend model | Profile |
| --- | --- | --- |
| `me_flow_rlap` | `ME-flow.rlap` | Single-frame low-latency state-space model |
| `facephys_rlap` | `FacePhys.rlap` | Default optimized state-space model |
| `me_chunk_rlap` | `ME-chunk.rlap` | Chunk state-space model |
| `rhythmmamba_rlap` | `RhythmMamba.rlap` | AAAI 2025 lightweight Mamba route |
| `physmamba_rlap` | `PhysMamba.rlap` | CCBR 2024 efficient Mamba route |
| `efficientphys_rlap` | `EfficientPhys.rlap` | Fast lightweight fallback baseline |

`me_flow_rlap` is the default because it is both recent and low-latency.

## Run smoke tests

Use the repository virtual environment:

```powershell
.\.venv\Scripts\python.exe .\rppg_model_deploy\smoke_test.py --models all
```

The test loads every model, checks required weight files, runs one synthetic
face-tensor inference chunk, and writes `smoke_results.json`.

## Swap a model

```python
from rppg_model_deploy import RppgModelAdapter

adapter = RppgModelAdapter("me_flow_rlap")
bvp = adapter.direct_bvp(faces_uint8)
metrics = adapter.process_faces(faces_uint8, fps=30.0)
```

To switch models, change only the key:

```python
adapter = RppgModelAdapter("rhythmmamba_rlap")
```

## Future external backends

The registry also tracks reference-only candidates such as RhythmJEPA, TYrPPG,
and Reperio-rPPG. They are intentionally not marked as deployed until their
checkpoints, licenses, dependencies, and runtime behavior are verified locally.
