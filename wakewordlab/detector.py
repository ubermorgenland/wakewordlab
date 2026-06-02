from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from wakewordlab.audio import AudioStream, TARGET_SR, iter_sliding_windows, load_audio_mono_16k
from wakewordlab.models import ModelInfo, _InferenceSession, resolve_model
from wakewordlab.vad import VAD


@dataclass
class DetectionEvent:
    wake_word: str
    confidence: float
    timestamp: float
    audio: np.ndarray   # 1s window that triggered detection, 16 kHz float32


class WakewordDetector:
    """
    On-device wake word detector.

    Usage — mic streaming:
        detector = WakewordDetector("hey_jarvis")

        @detector.on_detection
        def handle(event):
            print(event.wake_word, event.confidence)

        detector.start()
        detector.wait()   # block until Ctrl-C

    Usage — file scoring:
        score = detector.score(audio_array)
        result = detector.score_file("clip.wav")

    Usage — context manager:
        with WakewordDetector("hey_jarvis") as det:
            det.on_detection(handle)
            det.wait()
    """

    def __init__(
        self,
        model: str | Path,
        *,
        license_key: str | None = None,
        threshold: float | None = None,
        vad: bool = True,
        vad_threshold: float = 0.5,
        cooldown_sec: float = 1.5,
        window_sec: float = 1.0,
        stride_sec: float = 0.1,
        device: int | str | None = None,
        sample_rate: int = TARGET_SR,
    ) -> None:
        import dataclasses
        self._info = resolve_model(model)
        if license_key is not None:
            self._info = dataclasses.replace(self._info, license_key=license_key)
        self._session = _InferenceSession(self._info)

        self._threshold = threshold if threshold is not None else self._info.suggested_threshold
        self._cooldown_sec = cooldown_sec
        self._window_sec = window_sec
        self._stride_sec = stride_sec
        self._device = device
        self._sample_rate = sample_rate

        self._window_samples = int(round(window_sec * sample_rate))
        self._stride_samples = int(round(stride_sec * sample_rate))

        self._vad: VAD | None = VAD(threshold=vad_threshold, sample_rate=sample_rate) if vad else None
        self._callbacks: list[Callable[[DetectionEvent], None]] = []

        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=512)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    @property
    def wake_word(self) -> str:
        return self._info.wake_word

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = float(value)

    def on_detection(
        self, callback: Callable[[DetectionEvent], None]
    ) -> Callable[[DetectionEvent], None]:
        """Register a callback. Can be used as a decorator or called directly."""
        self._callbacks.append(callback)
        return callback

    def start(self) -> None:
        """Start mic streaming in a background thread (non-blocking)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="wakewordlab-detector")
        self._thread.start()

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it to exit."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def wait(self) -> None:
        """Block until stop() is called or KeyboardInterrupt."""
        try:
            while self._thread is not None and self._thread.is_alive():
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()

    def score(self, audio: np.ndarray) -> float:
        """Score a float32 mono array at 16 kHz. Returns probability 0–1."""
        if audio.size < self._window_samples:
            audio = np.pad(audio, (0, self._window_samples - audio.size))
        elif audio.size > self._window_samples:
            audio = audio[: self._window_samples]
        return self._session.run(audio)

    def score_file(self, path: str | Path) -> dict:
        """Score an audio file using a sliding window. Returns a summary dict."""
        audio = load_audio_mono_16k(path)
        results = [
            (t, self._session.run(w))
            for t, w in iter_sliding_windows(
                audio,
                window_sec=self._window_sec,
                stride_sec=self._stride_sec,
                sample_rate=self._sample_rate,
            )
        ]
        if not results:
            return {"path": str(path), "detected": False, "peak_score": 0.0, "peak_time_sec": 0.0}

        peak_time, peak_score = max(results, key=lambda x: x[1])
        return {
            "path": str(path),
            "wake_word": self.wake_word,
            "detected": peak_score >= self._threshold,
            "peak_score": float(peak_score),
            "peak_time_sec": float(peak_time),
            "hit_count": sum(1 for _, s in results if s >= self._threshold),
            "window_count": len(results),
            "threshold": self._threshold,
        }

    # ------------------------------------------------------------------ #
    # Context manager                                                      #
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "WakewordDetector":
        return self

    def __exit__(self, *args) -> None:
        self.stop()

    # ------------------------------------------------------------------ #
    # Internal streaming loop                                              #
    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        stream = AudioStream(
            device=self._device,
            sample_rate=self._sample_rate,
            block_size=512,
        )
        with stream:
            self._detection_loop(stream)

    def _detection_loop(self, stream: AudioStream) -> None:
        buffer: deque[float] = deque(maxlen=self._window_samples)
        samples_since_score = 0
        last_detection = 0.0

        while not self._stop_event.is_set():
            chunk = stream.get(timeout=0.05)
            if chunk is None:
                continue

            buffer.extend(chunk.tolist())
            samples_since_score += len(chunk)

            if len(buffer) < self._window_samples:
                continue
            if samples_since_score < self._stride_samples:
                continue

            samples_since_score = 0
            window = np.array(buffer, dtype=np.float32)

            if self._vad is not None and not self._vad.contains_speech(window):
                continue

            confidence = self._session.run(window)

            now = time.time()
            if confidence >= self._threshold and (now - last_detection) >= self._cooldown_sec:
                last_detection = now
                event = DetectionEvent(
                    wake_word=self.wake_word,
                    confidence=confidence,
                    timestamp=now,
                    audio=window.copy(),
                )
                for cb in self._callbacks:
                    try:
                        cb(event)
                    except Exception:
                        pass
