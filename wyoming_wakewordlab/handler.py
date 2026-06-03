import argparse
import logging
import time
from collections import deque

import numpy as np
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe
from wyoming.server import AsyncEventHandler
from wyoming.wake import Detection, NotDetected

_LOGGER = logging.getLogger(__name__)

_RATE = 16000
_WIDTH = 2   # 16-bit PCM
_CHANNELS = 1


class WakeWordHandler(AsyncEventHandler):
    def __init__(
        self,
        wyoming_info,
        cli_args: argparse.Namespace,
        sessions: dict,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._info = wyoming_info
        self._args = cli_args
        self._sessions = sessions

        self._converter = AudioChunkConverter(rate=_RATE, width=_WIDTH, channels=_CHANNELS)
        self._window_samples = int(cli_args.window_sec * _RATE)
        self._stride_samples = int(cli_args.stride_sec * _RATE)

        self._buffer: deque[float] = deque(maxlen=self._window_samples)
        self._samples_since_score = 0
        self._last_detection = 0.0
        self._detected = False

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self._info.event())
            return True

        if AudioStart.is_type(event.type):
            self._buffer.clear()
            self._samples_since_score = 0
            self._detected = False
            return True

        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            chunk = self._converter.convert(chunk)

            audio = np.frombuffer(chunk.audio, dtype=np.int16).astype(np.float32) / 32768.0
            self._buffer.extend(audio.tolist())
            self._samples_since_score += len(audio)

            if len(self._buffer) < self._window_samples:
                return True
            if self._samples_since_score < self._stride_samples:
                return True

            self._samples_since_score = 0
            now = time.monotonic()

            if (now - self._last_detection) < self._args.refractory_seconds:
                return True

            window = np.array(self._buffer, dtype=np.float32)

            for model_name, session in self._sessions.items():
                score = session.run(window)
                _LOGGER.debug("%s score=%.4f", model_name, score)

                if score >= self._args.threshold:
                    _LOGGER.info("Detection: %s (%.4f)", model_name, score)
                    self._last_detection = now
                    self._detected = True
                    await self.write_event(
                        Detection(name=model_name, timestamp=chunk.timestamp).event()
                    )

            return True

        if AudioStop.is_type(event.type):
            if not self._detected:
                await self.write_event(NotDetected().event())
            return True

        return True
