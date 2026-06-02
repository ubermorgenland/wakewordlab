# cython: language_level=3
"""
Compiled wake word loader.

Key reconstruction and all decryption logic live in compiled C.
No key constant appears in Python-level code or the Cython source.
_derive_key and _decrypt_wkw are cdef — not callable from Python.
WkwSession exposes only run(); the ORT session is a cdef attribute.
"""

import json
import struct
from pathlib import Path

import numpy as np
import onnxruntime as ort

cdef extern from "key.h":
    void wkw_get_public_key(unsigned char *out)

cdef bytes _MAGIC = b"WKW\x01"
cdef int _FINGERPRINT_LEN = 32
cdef int _NONCE_LEN = 12


cdef bytes _derive_key(license_key):
    cdef unsigned char key_buf[32]
    if license_key is None:
        wkw_get_public_key(key_buf)
        return bytes(key_buf[:32])

    # Commercial: HKDF(license_key, device_fingerprint)
    import uuid
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes as _hashes
    wkw_get_public_key(key_buf)   # zero out buffer first (unused here)
    device_salt = str(uuid.getnode()).encode()
    hkdf = HKDF(
        algorithm=_hashes.SHA256(),
        length=32,
        salt=device_salt,
        info=b"wakewordlab-v1",
    )
    raw = license_key.encode() if isinstance(license_key, str) else license_key
    return hkdf.derive(raw)


cdef bytes _decrypt_wkw(str path, bytes key):
    data = Path(path).read_bytes()
    offset = 0

    if data[offset:offset + 4] != _MAGIC:
        raise ValueError(f"Not a valid .wkw file: {path}")
    offset += 4
    offset += _FINGERPRINT_LEN

    nonce = data[offset:offset + _NONCE_LEN]
    offset += _NONCE_LEN

    ct_len = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    ct_and_tag = data[offset:offset + ct_len]

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    try:
        return AESGCM(key).decrypt(nonce, ct_and_tag, None)
    except Exception:
        raise ValueError("Decryption failed — wrong key or corrupted file")


cdef dict _read_wkw_metadata(str path):
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


cdef class WkwSession:
    cdef object _session
    cdef str _input_name
    cdef str _output_name
    cdef public dict metadata

    def __cinit__(self, str wkw_path, license_key=None):
        cdef unsigned char[:] _dummy  # suppress unused warning
        cdef bytes key = _derive_key(license_key)
        cdef bytes model_bytes = _decrypt_wkw(wkw_path, key)

        self.metadata = _read_wkw_metadata(wkw_path)

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        try:
            self._session = ort.InferenceSession(
                model_bytes,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
        finally:
            del model_bytes

    def run(self, audio):
        x = np.asarray(audio, dtype=np.float32).reshape(1, -1)
        out = self._session.run([self._output_name], {self._input_name: x})[0]
        logits = np.asarray(out, dtype=np.float32).reshape(-1)
        if logits.size > 1:
            exps = np.exp(logits - logits.max())
            return float((exps / exps.sum())[1])
        return float(1.0 / (1.0 + np.exp(-logits[0])))
