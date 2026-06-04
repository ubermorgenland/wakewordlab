# wakewordlab

On-device wake word detection for Python.

Includes a [Silero VAD](https://github.com/snakers4/silero-vad) pre-filter that gates inference on speech frames only, reducing CPU usage on silence and cutting false positives.

## Installation

```bash
pip install wakewordlab          # file scoring only
pip install wakewordlab[mic]     # + microphone streaming
```

## Quick start

```python
import wakewordlab

# Download a model (cached to ~/.cache/wakewordlab/models/)
wakewordlab.download("hey_jarvis")

# Start listening — VAD is on by default
detector = wakewordlab.WakewordDetector("hey_jarvis")

@detector.on_detection
def handle(event):
    print(f"Detected: {event.wake_word}  confidence={event.confidence:.2f}")

detector.start()
detector.wait()   # blocks until Ctrl-C
```

## Streaming with VAD

VAD ([Silero v5](https://github.com/snakers4/silero-vad)) is enabled by default. It gates the wake word model so inference only runs on frames that contain speech, keeping CPU usage low during silence.

```python
detector = wakewordlab.WakewordDetector(
    "hey_jarvis",
    vad=True,            # default — Silero VAD pre-filter
    vad_threshold=0.5,   # lower = more sensitive (pass more frames)
    threshold=0.5,       # wake word detection threshold
    cooldown_sec=1.5,    # min seconds between consecutive detections
    stride_sec=0.1,      # how often a window is scored (seconds)
)

@detector.on_detection
def handle(event):
    print(f"{event.wake_word}  confidence={event.confidence:.2f}")
    # event.audio — the 1s window that triggered (16 kHz float32 numpy array)

with detector:
    detector.wait()
```

To disable VAD (score every window regardless of speech):

```python
detector = wakewordlab.WakewordDetector("hey_jarvis", vad=False)
```

## Score an audio file

```python
result = detector.score_file("clip.wav")
# {
#   "detected": True,
#   "peak_score": 0.92,
#   "peak_time_sec": 0.4,
#   "hit_count": 3,
#   "window_count": 21,
#   "threshold": 0.5,
# }
```

## Score a raw array

```python
import numpy as np
audio = np.zeros(16000, dtype=np.float32)   # 1 s at 16 kHz
prob = detector.score(audio)                 # float 0–1
```

## List available models

```python
wakewordlab.list_models()   # → ["hey_jarvis", ...]
```

## Options

| Parameter       | Default | Description                                      |
|-----------------|---------|--------------------------------------------------|
| `threshold`     | `0.5`   | Wake word detection threshold (0–1)              |
| `vad`           | `True`  | Enable Silero VAD pre-filter                     |
| `vad_threshold` | `0.5`   | VAD sensitivity — lower passes more frames       |
| `cooldown_sec`  | `1.5`   | Minimum seconds between consecutive detections   |
| `window_sec`    | `1.0`   | Scoring window length in seconds                 |
| `stride_sec`    | `0.1`   | Hop between windows in seconds                   |
| `device`        | `None`  | Mic device index (default system mic)            |

## List audio input devices

```python
from wakewordlab.audio import AudioStream
print(AudioStream.list_devices())
```

## Home Assistant (Wyoming Protocol)

wakewordlab can run as a [Wyoming](https://github.com/rhasspy/wyoming) wake word server for [Home Assistant](https://www.home-assistant.io/), replacing or supplementing the built-in wake word engines.

### Setup

**1. Create `docker-compose.yml`:**

```yaml
services:
  wyoming-wakewordlab:
    image: ubermorgenai/wyoming-wakewordlab:latest
    restart: unless-stopped
    ports:
      - "10400:10400"
    volumes:
      - model-cache:/root/.cache/wakewordlab
    command: >
      --uri tcp://0.0.0.0:10400
      --models hey_jarvis
      --threshold 0.6
      --refractory-seconds 2.0

volumes:
  model-cache:
```

**2. Start the server:**

```bash
docker compose up -d
```

The model downloads automatically on first start and is cached in the volume — subsequent restarts are instant.

**3. Add to Home Assistant:**

Settings → Devices & Services → Add Integration → **Wyoming Protocol**
- Host: IP address of the machine running Docker
- Port: `10400`

Home Assistant will detect the available wake words and you can assign one to a voice assistant pipeline.

### Multiple models

Run several wake words simultaneously by listing them:

```yaml
command: >
  --uri tcp://0.0.0.0:10400
  --models hey_jarvis stop
  --threshold 0.6
```

---

## Commercial models

Commercial models are distributed as `.wkw` files with a license key:

```python
detector = wakewordlab.WakewordDetector(
    "path/to/custom.wkw",
    license_key="xxxx-xxxx-xxxx-xxxx",
)
```

## License

Public models are licensed for **non-commercial use only**.
