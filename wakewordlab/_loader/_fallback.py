"""
Pure-Python fallback for when the Cython extension is not compiled.
FOR DEVELOPMENT ONLY — the key is visible in Python bytecode here.
Never use this in production distribution.
"""

import hashlib
import json
import struct
import warnings
from pathlib import Path

import numpy as np
import onnxruntime as ort
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

warnings.warn(
    "wakewordlab: Cython _loader extension not found. "
    "Running in unprotected fallback mode (development only). "
    "Build the extension with: pip install cython && python setup.py build_ext --inplace",
    stacklevel=3,
)

_PUBLIC_KEY = bytes.fromhex(
    "9f3e7c1a5b8d2e4f6a0c9b3e7d1f5a2c"
    "4b8e0d6a2f9c3e7b5a1d8f4c2e6b9d0f"
)
_MAGIC = b"WKW\x01"
_FINGERPRINT_LEN = 32
_NONCE_LEN = 12


def _derive_key(license_key) -> bytes:
    if license_key is None:
        return _PUBLIC_KEY
    import uuid
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes
    device_salt = str(uuid.getnode()).encode()
    hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=device_salt, info=b"wakewordlab-v1")
    raw = license_key.encode() if isinstance(license_key, str) else license_key
    return hkdf.derive(raw)


def _decrypt_wkw(path: str, key: bytes) -> bytes:
    data = Path(path).read_bytes()
    offset = 0
    if data[offset:offset + 4] != _MAGIC:
        raise ValueError(f"Not a valid .wkw file: {path}")
    offset += 4 + _FINGERPRINT_LEN
    nonce = data[offset:offset + _NONCE_LEN]
    offset += _NONCE_LEN
    ct_len = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    ct_and_tag = data[offset:offset + ct_len]
    try:
        return AESGCM(key).decrypt(nonce, ct_and_tag, None)
    except Exception:
        raise ValueError("Decryption failed — wrong key or corrupted file")


def _read_wkw_metadata(path: str) -> dict:
    data = Path(path).read_bytes()
    offset = 4 + _FINGERPRINT_LEN + _NONCE_LEN
    ct_len = struct.unpack_from("<I", data, offset)[0]
    offset += 4 + ct_len
    meta_len = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    try:
        return json.loads(data[offset:offset + meta_len])
    except Exception:
        return {}


class WkwSession:
    def __init__(self, wkw_path: str, license_key=None):
        key = _derive_key(license_key)
        model_bytes = _decrypt_wkw(wkw_path, key)
        self.metadata = _read_wkw_metadata(wkw_path)
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            model_bytes, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    def run(self, audio) -> float:
        x = np.asarray(audio, dtype=np.float32).reshape(1, -1)
        out = self._session.run([self._output_name], {self._input_name: x})[0]
        logits = np.asarray(out, dtype=np.float32).reshape(-1)
        if logits.size > 1:
            exps = np.exp(logits - logits.max())
            return float((exps / exps.sum())[1])
        return float(1.0 / (1.0 + np.exp(-logits[0])))
