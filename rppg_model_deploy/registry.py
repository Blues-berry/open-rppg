from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_KEY = "me_flow_rlap"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    backend: str
    model_name: str
    status: str
    profile: str
    input_shape: Tuple[int, int, int, int]
    weights: Tuple[str, ...]
    reference: str
    reason: str
    notes: str = ""

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["input_shape"] = list(self.input_shape)
        data["weights"] = list(self.weights)
        return data


_DEPLOYED_MODELS: Dict[str, ModelSpec] = {
    "me_flow_rlap": ModelSpec(
        key="me_flow_rlap",
        display_name="ME-flow.rlap",
        backend="open-rppg bundled",
        model_name="ME-flow.rlap",
        status="deployed",
        profile="single-frame low-latency state-space model",
        input_shape=(1, 36, 36, 3),
        weights=("rppg/weights/ME.rlap.weights.h5", "rppg/weights/state.pkl"),
        reference="https://arxiv.org/abs/2504.01774",
        reason="Newest bundled low-latency state-space route from the open-rppg author.",
    ),
    "facephys_rlap": ModelSpec(
        key="facephys_rlap",
        display_name="FacePhys.rlap",
        backend="open-rppg bundled",
        model_name="FacePhys.rlap",
        status="deployed",
        profile="single-frame optimized state-space model",
        input_shape=(1, 36, 36, 3),
        weights=("rppg/weights/FacePhys.rlap.weights.h5",),
        reference="https://github.com/KegangWangCCNU/open-rppg",
        reason="Default robust model in current open-rppg release.",
    ),
    "me_chunk_rlap": ModelSpec(
        key="me_chunk_rlap",
        display_name="ME-chunk.rlap",
        backend="open-rppg bundled",
        model_name="ME-chunk.rlap",
        status="deployed",
        profile="chunk state-space model",
        input_shape=(160, 36, 36, 3),
        weights=("rppg/weights/ME.rlap.weights.h5", "rppg/weights/state.pkl"),
        reference="https://arxiv.org/abs/2504.01774",
        reason="Chunk inference variant for better throughput when latency is less critical.",
    ),
    "rhythmmamba_rlap": ModelSpec(
        key="rhythmmamba_rlap",
        display_name="RhythmMamba.rlap",
        backend="open-rppg bundled",
        model_name="RhythmMamba.rlap",
        status="deployed",
        profile="AAAI 2025 lightweight Mamba",
        input_shape=(160, 128, 128, 3),
        weights=("rppg/weights/RhythmMamba.rlap.weights.h5",),
        reference="https://ojs.aaai.org/index.php/AAAI/article/view/33082",
        reason="Fast and lightweight Mamba-family rPPG model included in the toolbox.",
    ),
    "physmamba_rlap": ModelSpec(
        key="physmamba_rlap",
        display_name="PhysMamba.rlap",
        backend="open-rppg bundled",
        model_name="PhysMamba.rlap",
        status="deployed",
        profile="CCBR 2024 efficient Mamba",
        input_shape=(128, 128, 128, 3),
        weights=("rppg/weights/PhysMamba.rlap.weights.h5",),
        reference="https://link.springer.com/chapter/10.1007/978-981-96-1903-7_22",
        reason="Efficient dual-branch Mamba architecture for remote physiological measurement.",
    ),
    "efficientphys_rlap": ModelSpec(
        key="efficientphys_rlap",
        display_name="EfficientPhys.rlap",
        backend="open-rppg bundled",
        model_name="EfficientPhys.rlap",
        status="deployed",
        profile="fast convolutional baseline",
        input_shape=(160, 72, 72, 3),
        weights=("rppg/weights/EfficientPhys.rlap.weights.h5",),
        reference="https://openaccess.thecvf.com/content/WACV2023/html/Liu_EfficientPhys_Enabling_Simple_Fast_and_Accurate_Camera-Based_Cardiac_Measurement_WACV_2023_paper.html",
        reason="Small fast baseline that is useful as a fallback during model swaps.",
    ),
}

DEPLOYED_MODEL_KEYS = tuple(_DEPLOYED_MODELS.keys())

EXTERNAL_CANDIDATES = {
    "rhythmjepa": {
        "status": "reference_only",
        "reference": "https://arxiv.org/abs/2606.31736",
        "repository": "https://github.com/deconasser/RhythmJEPA",
        "notes": "Very recent 2026 rPPG paper. Keep as a future backend after weights and runtime are verified.",
    },
    "tyrppg": {
        "status": "reference_only",
        "reference": "https://arxiv.org/abs/2511.05833",
        "repository": "https://github.com/Taixi-CHEN/TYrPPG",
        "notes": "2025 lightweight Mambaout-style route. Needs separate dependency and checkpoint audit.",
    },
    "reperio_rppg": {
        "status": "reference_only",
        "reference": "https://arxiv.org/abs/2511.05946",
        "repository": "https://github.com/deconasser/Reperio-rPPG",
        "notes": "2025 SOTA-oriented graph/transformer route; likely less lightweight than the deployed set.",
    },
}


def _canonical_key(name: str) -> str:
    return name.strip().replace("-", "_").replace(".", "_").lower()


def list_deployed_models() -> List[ModelSpec]:
    return list(_DEPLOYED_MODELS.values())


def get_model_spec(key_or_name: str) -> ModelSpec:
    if key_or_name in _DEPLOYED_MODELS:
        return _DEPLOYED_MODELS[key_or_name]

    canonical = _canonical_key(key_or_name)
    for key, spec in _DEPLOYED_MODELS.items():
        if canonical in {key, _canonical_key(spec.model_name), _canonical_key(spec.display_name)}:
            return spec
    raise KeyError(
        "Unknown rPPG model {!r}. Available: {}".format(
            key_or_name, ", ".join(DEPLOYED_MODEL_KEYS)
        )
    )


def model_specs_as_dict(keys: Iterable[str] = DEPLOYED_MODEL_KEYS) -> Dict[str, Dict[str, object]]:
    return {get_model_spec(key).key: get_model_spec(key).to_dict() for key in keys}


def check_assets(key_or_name: str) -> Dict[str, object]:
    spec = get_model_spec(key_or_name)
    files = []
    ok = True
    for rel_path in spec.weights:
        path = REPO_ROOT / rel_path
        exists = path.exists()
        files.append(
            {
                "path": str(path),
                "exists": exists,
                "size_bytes": path.stat().st_size if exists else None,
            }
        )
        ok = ok and exists
    return {"model": spec.key, "ok": ok, "files": files}
