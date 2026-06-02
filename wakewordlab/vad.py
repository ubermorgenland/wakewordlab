from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort

VAD_CACHE_DIR = Path.home() / ".cache" / "wakewordlab" / "vad"
_VAD_MODEL_PATH = VAD_CACHE_DIR / "silero_vad.onnx"

# Silero VAD v5 — pinned release
_VAD_MODEL_URL = (
    "https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx"
)

# 512 samples = 32 ms at 16 kHz (Silero v5 requirement)
_CHUNK_SIZE = 512
_STATE_SHAPE = (2, 1, 128)  # v5: single combined state tensor [2, batch, 128]


def _ensure_vad_model() -> Path:
    if _VAD_MODEL_PATH.exists():
        return _VAD_MODEL_PATH
    VAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _VAD_MODEL_PATH.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(_VAD_MODEL_URL, tmp)
        tmp.rename(_VAD_MODEL_PATH)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return _VAD_MODEL_PATH


class VAD:
    """
    Silero VAD v5 wrapper.
    Stateful — maintains LSTM state across consecutive calls for streaming use.
    Call reset() to clear state between unrelated audio segments.
    """

    def __init__(self, threshold: float = 0.5, sample_rate: int = 16000) -> None:
        if sample_rate not in (8000, 16000):
            raise ValueError("Silero VAD supports sample_rate 8000 or 16000 only")
        self.threshold = threshold
        self._sr = sample_rate
        model_path = _ensure_vad_model()
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._output_names = [o.name for o in self._session.get_outputs()]
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros(_STATE_SHAPE, dtype=np.float32)

    def is_speech(self, chunk: np.ndarray) -> bool:
        """
        Score a single audio chunk (exactly 512 samples at 16 kHz).
        Updates LSTM state in place for streaming continuity.
        """
        if chunk.size != _CHUNK_SIZE:
            raise ValueError(f"VAD expects {_CHUNK_SIZE} samples, got {chunk.size}")

        feed = {
            "input": chunk.astype(np.float32).reshape(1, _CHUNK_SIZE),
            "state": self._state,
            "sr": np.array(self._sr, dtype=np.int64),
        }

        outputs = self._session.run(self._output_names, feed)
        prob = float(outputs[0].squeeze())
        self._state = outputs[1]  # stateN

        return prob >= self.threshold

    def contains_speech(self, audio: np.ndarray) -> bool:
        """
        Return True if any 512-sample chunk in audio contains speech.
        Runs with a temporary fresh state so persistent streaming state is unaffected.
        """
        saved = self._state.copy()
        self.reset()
        try:
            n = len(audio)
            for start in range(0, n - _CHUNK_SIZE + 1, _CHUNK_SIZE):
                if self.is_speech(audio[start : start + _CHUNK_SIZE]):
                    return True
            return False
        finally:
            self._state = saved
