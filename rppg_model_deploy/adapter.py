import sys
from pathlib import Path
from typing import Optional

import numpy as np

from .registry import DEFAULT_MODEL_KEY, REPO_ROOT, ModelSpec, get_model_spec


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class RppgModelAdapter:
    """Small stable wrapper around the model backend used by application code."""

    def __init__(self, model_key: str = DEFAULT_MODEL_KEY):
        self.spec: ModelSpec = get_model_spec(model_key)
        self._model = None
        self._state = None

    @property
    def model(self):
        if self._model is None:
            self.load()
        return self._model

    def load(self):
        if self._model is None:
            import rppg

            self._model = rppg.Model(self.spec.model_name)
            self._state = self._model.state
        return self._model

    def direct_bvp(self, faces_uint8: np.ndarray) -> np.ndarray:
        """Run one backend-sized chunk and return the raw BVP output."""
        model = self.model
        batch = self._fit_faces(faces_uint8, model.input)
        output, self._state = model.call(batch, self._state)
        return np.asarray(output["bvp"], dtype=np.float32)

    def process_faces(self, faces_uint8: np.ndarray, fps: float = 30.0) -> Optional[dict]:
        """Use open-rppg's complete face-tensor path and return HR/SQI metrics."""
        faces_uint8 = self._validate_faces(faces_uint8)
        return self.model.process_faces_tensor(faces_uint8, fps=fps)

    @staticmethod
    def _validate_faces(faces_uint8: np.ndarray) -> np.ndarray:
        faces = np.asarray(faces_uint8)
        if faces.dtype != np.uint8:
            raise TypeError("faces_uint8 must have dtype uint8")
        if faces.ndim != 4 or faces.shape[-1] != 3:
            raise TypeError("faces_uint8 must have shape (frames, height, width, 3)")
        if len(faces) == 0:
            raise ValueError("faces_uint8 must contain at least one frame")
        return faces

    @classmethod
    def _fit_faces(cls, faces_uint8: np.ndarray, target_shape) -> np.ndarray:
        faces = cls._validate_faces(faces_uint8)
        frames, height, width, channels = target_shape
        if channels != 3:
            raise ValueError("Only RGB input shapes are supported")

        if faces.shape[1:3] != (height, width):
            import cv2

            faces = np.stack(
                [
                    cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                    for frame in faces
                ],
                axis=0,
            ).astype(np.uint8, copy=False)

        if len(faces) < frames:
            pad = np.repeat(faces[-1:], frames - len(faces), axis=0)
            faces = np.concatenate([faces, pad], axis=0)
        elif len(faces) > frames:
            faces = faces[:frames]

        return faces.astype(np.uint8, copy=False)
