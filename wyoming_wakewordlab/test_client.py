"""
Simple Wyoming test client — sends a WAV file or silence to the server
and prints any detection events.

Usage:
    python test_client.py                          # send silence
    python test_client.py --wav /path/to/clip.wav  # send a wav file
    python test_client.py --host 192.168.1.x       # remote host
"""

import argparse
import asyncio
import struct
import sys

import numpy as np

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.info import Describe
from wyoming.wake import Detection, NotDetected

RATE = 16000
WIDTH = 2
CHANNELS = 1
CHUNK_SAMPLES = 1024


def _load_wav(path: str) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != RATE:
        from math import gcd
        g = gcd(RATE, sr)
        audio = resample_poly(audio, RATE // g, sr // g).astype(np.float32)
    return audio


def _to_pcm(audio: np.ndarray) -> bytes:
    return (audio * 32767).astype(np.int16).tobytes()


async def run(args: argparse.Namespace) -> None:
    if args.wav:
        audio = _load_wav(args.wav)
        print(f"Loaded {args.wav}: {len(audio)/RATE:.1f}s")
    else:
        # 3 seconds of silence
        audio = np.zeros(RATE * 3, dtype=np.float32)
        print("Sending 3 seconds of silence")

    async with AsyncTcpClient(args.host, args.port) as client:
        # Query server info
        await client.write_event(Describe().event())
        info = await client.read_event()
        print(f"Server: {info}")

        # Stream audio
        await client.write_event(
            AudioStart(rate=RATE, width=WIDTH, channels=CHANNELS).event()
        )

        pcm = _to_pcm(audio)
        chunk_bytes = CHUNK_SAMPLES * WIDTH
        for i in range(0, len(pcm), chunk_bytes):
            chunk = pcm[i : i + chunk_bytes]
            await client.write_event(
                AudioChunk(
                    rate=RATE, width=WIDTH, channels=CHANNELS, audio=chunk
                ).event()
            )

        await client.write_event(AudioStop().event())

        # Wait for response
        response = await client.read_event()
        if Detection.is_type(response.type):
            det = Detection.from_event(response)
            print(f"\n>>> DETECTED: {det.name} at {det.timestamp}ms")
        elif NotDetected.is_type(response.type):
            print("\n>>> Not detected")
        else:
            print(f"\n>>> Response: {response}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=10400)
    parser.add_argument("--wav", default=None, help="WAV file to send")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
