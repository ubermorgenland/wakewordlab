from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort

CACHE_DIR = Path.home() / ".cache" / "wakewordlab" / "models"
_REGISTRY_CACHE_PATH = Path.home() / ".cache" / "wakewordlab" / "registry.json"
_REGISTRY_URL = "https://storage.googleapis.com/wakewordlab-models/registry.json"
_REGISTRY_TTL = 3600  # seconds before re-fetching

# Runtime-registered local paths (testing / custom models)
_local_registry: dict[str, tuple[Path, str]] = {}


@dataclass
class ModelInfo:
    slug: str
    path: Path
    wake_word: str
    sample_rate: int = 16000
    suggested_threshold: float = 0.5
    license_key: str | None = None


def _fetch_registry() -> dict:
    """Return registry dict, using a local cache with TTL. Silent on network errors."""
    import time
    if _REGISTRY_CACHE_PATH.exists():
        age = time.time() - _REGISTRY_CACHE_PATH.stat().st_mtime
        if age < _REGISTRY_TTL:
            try:
                return json.loads(_REGISTRY_CACHE_PATH.read_text())
            except Exception:
                pass
    try:
        import requests
        resp = requests.get(_REGISTRY_URL, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        _REGISTRY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _REGISTRY_CACHE_PATH.write_text(json.dumps(data))
        return data
    except Exception:
        # Fall back to stale cache rather than failing hard
        if _REGISTRY_CACHE_PATH.exists():
            try:
                return json.loads(_REGISTRY_CACHE_PATH.read_text())
            except Exception:
                pass
        return {}


def list_models() -> list[str]:
    """List all available public models (local + remote registry)."""
    return sorted(set(_fetch_registry()) | set(_local_registry))


def register_local(slug: str, path: str | Path, *, wake_word: str | None = None) -> None:
    """Register a local .onnx / .wkw file or model directory under a slug."""
    p = Path(path).expanduser().resolve()
    if p.is_dir():
        candidate = p / "model-best.onnx"
        if not candidate.exists():
            raise FileNotFoundError(f"No model-best.onnx found in {p}")
        p = candidate
    if not p.exists():
        raise FileNotFoundError(p)
    _local_registry[slug] = (p, wake_word or slug)


def download(slug: str, *, force: bool = False) -> Path:
    """Download a public model to the local cache and return the cached .wkw path."""
    if slug in _local_registry:
        return _local_registry[slug][0]

    registry = _fetch_registry()
    if slug not in registry:
        available = list(registry.keys())
        hint = (
            f"\nRegister a local file: wakewordlab.register_local({slug!r}, '/path/to/model.wkw')"
            if not available
            else f"\nAvailable: {available}"
        )
        raise ValueError(f"Unknown model {slug!r}.{hint}")

    entry = registry[slug]
    cached = CACHE_DIR / slug / f"{slug}.wkw"

    if cached.exists() and not force:
        return cached

    cached.parent.mkdir(parents=True, exist_ok=True)

    import requests
    print(f"Downloading {slug} ({entry['size_bytes']:,} bytes)…")
    resp = requests.get(entry["url"], stream=True, timeout=60)
    resp.raise_for_status()

    tmp = cached.with_suffix(".tmp")
    try:
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                fh.write(chunk)

        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise RuntimeError(f"Checksum mismatch for {slug!r}: got {digest}")

        shutil.move(str(tmp), str(cached))
    finally:
        tmp.unlink(missing_ok=True)

    return cached


def get_model_path(slug: str) -> tuple[Path, str]:
    """Return (path, wake_word) for a slug, downloading if necessary."""
    if slug in _local_registry:
        return _local_registry[slug]

    cached = CACHE_DIR / slug / f"{slug}.wkw"
    if not cached.exists():
        download(slug)

    registry = _fetch_registry()
    entry = registry.get(slug, {})
    return cached, entry.get("wake_word", slug)


def resolve_model(model: str | Path) -> ModelInfo:
    """
    Accept a slug, a .wkw path, a .onnx path, or a directory containing model-best.onnx.
    Returns a ModelInfo with resolved path and metadata.
    """
    p = Path(model).expanduser()

    # Directory containing model-best.onnx
    if p.is_dir():
        candidate = p / "model-best.onnx"
        if not candidate.exists():
            raise FileNotFoundError(f"No model-best.onnx in {p}")
        meta = _load_metadata(p)
        return ModelInfo(
            slug=p.name,
            path=candidate,
            wake_word=meta.get("wake_word", p.name),
            sample_rate=int(meta.get("sample_rate", 16000)),
            suggested_threshold=float(meta.get("threshold", 0.5)),
        )

    # Explicit .wkw path
    if p.suffix == ".wkw":
        if not p.exists():
            raise FileNotFoundError(p)
        meta = _read_wkw_metadata(p)
        return ModelInfo(
            slug=p.stem,
            path=p,
            wake_word=meta.get("wake_word", p.stem),
            sample_rate=int(meta.get("sample_rate", 16000)),
            suggested_threshold=float(meta.get("threshold", 0.5)),
        )

    # Explicit .onnx path
    if p.suffix == ".onnx":
        if not p.exists():
            raise FileNotFoundError(p)
        meta = _load_metadata(p.parent)
        return ModelInfo(
            slug=p.stem,
            path=p,
            wake_word=meta.get("wake_word", p.stem),
            sample_rate=int(meta.get("sample_rate", 16000)),
            suggested_threshold=float(meta.get("threshold", 0.5)),
        )

    # Treat as slug
    slug = str(model)
    path, wake_word = get_model_path(slug)
    return ModelInfo(slug=slug, path=path, wake_word=wake_word)


def _read_wkw_metadata(path: Path) -> dict:
    """Read plaintext JSON metadata from end of a .wkw file without decrypting."""
    import struct
    data = path.read_bytes()
    _FINGERPRINT_LEN = 32
    _NONCE_LEN = 12
    offset = 4 + _FINGERPRINT_LEN + _NONCE_LEN
    ct_len = struct.unpack_from("<I", data, offset)[0]
    offset += 4 + ct_len
    meta_len = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    try:
        return json.loads(data[offset:offset + meta_len])
    except Exception:
        return {}


def _load_metadata(directory: Path) -> dict:
    for name in ("inference_config.json", "metadata.json", "results.json"):
        candidate = directory / name
        if candidate.exists():
            try:
                return json.loads(candidate.read_text())
            except Exception:
                pass
    return {}


class _InferenceSession:
    """
    Unified inference session.
    .wkw files → WkwSession (Cython compiled loader, or Python fallback).
    .onnx files → plain onnxruntime (dev/testing only, no protection).
    """

    def __init__(self, info: ModelInfo) -> None:
        self.info = info
        if info.path.suffix == ".wkw":
            from wakewordlab._loader import WkwSession
            self._impl = WkwSession(str(info.path), info.license_key)
            self._wkw = True
        else:
            self._wkw = False
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            self._session = ort.InferenceSession(
                str(info.path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name

    def run(self, audio: np.ndarray) -> float:
        """Score a float32 mono array at 16 kHz. Returns probability 0–1."""
        if self._wkw:
            return self._impl.run(audio)
        x = audio.astype(np.float32).reshape(1, -1)
        out = self._session.run([self._output_name], {self._input_name: x})[0]
        logits = np.asarray(out, dtype=np.float32).reshape(-1)
        if logits.size > 1:
            exps = np.exp(logits - logits.max())
            return float((exps / exps.sum())[1])
        return float(1.0 / (1.0 + np.exp(-logits[0])))
