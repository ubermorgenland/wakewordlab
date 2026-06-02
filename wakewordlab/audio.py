from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000

try:
    import sounddevice as sd
    _HAS_SOUNDDEVICE = True
except ImportError:
    _HAS_SOUNDDEVICE = False


def load_audio_file(path: str | Path) -> tuple[np.ndarray, int]:
    """Load any audio file, return (float32 mono array, sample_rate)."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), sr


def resample_to_16k(audio: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == TARGET_SR:
        return audio.astype(np.float32)
    from math import gcd
    g = gcd(TARGET_SR, src_sr)
    return resample_poly(audio.astype(np.float32), TARGET_SR // g, src_sr // g).astype(np.float32)


def load_audio_mono_16k(path: str | Path) -> np.ndarray:
    audio, sr = load_audio_file(path)
    return resample_to_16k(audio, sr)


def iter_sliding_windows(
    audio: np.ndarray,
    *,
    window_sec: float = 1.0,
    stride_sec: float = 0.1,
    sample_rate: int = TARGET_SR,
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (start_sec, window) for a sliding window over audio."""
    win = max(1, int(round(window_sec * sample_rate)))
    hop = max(1, int(round(stride_sec * sample_rate)))

    if audio.size <= win:
        yield 0.0, np.pad(audio, (0, win - audio.size)).astype(np.float32)
        return

    start = 0
    while start < audio.size:
        chunk = audio[start : start + win]
        if chunk.size < win:
            chunk = np.pad(chunk, (0, win - chunk.size))
        yield start / float(sample_rate), chunk.astype(np.float32)
        if start + win >= audio.size:
            break
        start += hop


class AudioStream:
    """
    Non-blocking microphone capture.
    Puts float32 mono chunks into an internal queue consumed by the caller.

    Requires the `mic` optional dependency: pip install wakewordlab[mic]
    """

    def __init__(
        self,
        *,
        device: int | str | None = None,
        sample_rate: int = TARGET_SR,
        block_size: int = 512,
    ) -> None:
        if not _HAS_SOUNDDEVICE:
            raise ImportError(
                "Microphone capture requires sounddevice.\n"
                "Install it with: pip install wakewordlab[mic]"
            )
        self.sample_rate = sample_rate
        self.block_size = block_size
        self._device = device
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=256)
        self._stream: sd.InputStream | None = None

    def start(self) -> None:
        self._stream = sd.InputStream(
            device=self._device,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get(self, timeout: float = 0.05) -> np.ndarray | None:
        """Return next audio chunk or None on timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _callback(self, indata, frames, time_info, status):
        chunk = indata[:, 0].copy()
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            pass  # drop oldest implicitly — queue is bounded

    @staticmethod
    def list_devices() -> list[dict]:
        if not _HAS_SOUNDDEVICE:
            raise ImportError("sounddevice not installed")
        devices = sd.query_devices()
        return [
            {"index": i, "name": d["name"], "input_channels": d["max_input_channels"]}
            for i, d in enumerate(devices)
            if d["max_input_channels"] > 0
        ]

    def __enter__(self) -> "AudioStream":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()
