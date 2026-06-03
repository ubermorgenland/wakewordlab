"""
Wyoming test client.

Usage:
    python test_client.py                          # send 3s silence
    python test_client.py --wav /path/to/clip.wav  # send a wav file
    python test_client.py --live                   # stream from mic (Ctrl-C to stop)
    python test_client.py --host 192.168.1.x       # remote host
"""

import argparse
import asyncio
import queue
import sys
import threading

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
    from math import gcd
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != RATE:
        g = gcd(RATE, sr)
        audio = resample_poly(audio, RATE // g, sr // g).astype(np.float32)
    return audio


def _to_pcm(audio: np.ndarray) -> bytes:
    return (audio * 32767).astype(np.int16).tobytes()


async def _send_static(client, audio: np.ndarray) -> None:
    await client.write_event(AudioStart(rate=RATE, width=WIDTH, channels=CHANNELS).event())
    pcm = _to_pcm(audio)
    chunk_bytes = CHUNK_SAMPLES * WIDTH
    for i in range(0, len(pcm), chunk_bytes):
        await client.write_event(
            AudioChunk(rate=RATE, width=WIDTH, channels=CHANNELS, audio=pcm[i:i+chunk_bytes]).event()
        )
    await client.write_event(AudioStop().event())
    response = await client.read_event()
    if response is None:
        print("\n>>> No response")
    elif Detection.is_type(response.type):
        det = Detection.from_event(response)
        print(f"\n>>> DETECTED: {det.name}")
    elif NotDetected.is_type(response.type):
        print("\n>>> Not detected")


async def _send_live(client) -> None:
    import sounddevice as sd

    audio_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    def callback(indata, frames, time_info, status):
        audio_queue.put(indata[:, 0].copy())

    print("Listening... Ctrl-C to stop\n")
    await client.write_event(AudioStart(rate=RATE, width=WIDTH, channels=CHANNELS).event())

    stream = sd.InputStream(samplerate=RATE, channels=1, blocksize=CHUNK_SAMPLES,
                             dtype="float32", callback=callback)

    # Read detections from server in background
    async def read_detections():
        while True:
            event = await client.read_event()
            if event is None:
                break
            if Detection.is_type(event.type):
                det = Detection.from_event(event)
                print(f"\n>>> DETECTED: {det.name}", flush=True)

    reader_task = asyncio.create_task(read_detections())

    try:
        with stream:
            while True:
                try:
                    chunk = audio_queue.get(timeout=0.1)
                    pcm = _to_pcm(chunk)
                    await client.write_event(
                        AudioChunk(rate=RATE, width=WIDTH, channels=CHANNELS, audio=pcm).event()
                    )
                except queue.Empty:
                    await asyncio.sleep(0)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        reader_task.cancel()
        await client.write_event(AudioStop().event())


async def run(args: argparse.Namespace) -> None:
    client = AsyncTcpClient(args.host, args.port)
    await client.connect()
    try:
        await client.write_event(Describe().event())
        info = await client.read_event()
        print(f"Server: {info}\n")

        if args.live:
            await _send_live(client)
        elif args.wav:
            audio = _load_wav(args.wav)
            print(f"Loaded {args.wav}: {len(audio)/RATE:.1f}s")
            await _send_static(client, audio)
        else:
            print("Sending 3 seconds of silence")
            await _send_static(client, np.zeros(RATE * 3, dtype=np.float32))
    finally:
        try:
            await client.disconnect()
        except (ConnectionResetError, OSError):
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=10400)
    parser.add_argument("--wav", default=None, help="WAV file to send")
    parser.add_argument("--live", action="store_true", help="Stream from mic")
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
