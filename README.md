# wakewordlab

On-device wake word detection for Python.

Includes a [Silero VAD](https://github.com/snakers4/silero-vad) pre-filter that gates inference on speech frames only, reducing CPU usage on silence and cutting false positives.

## Why use wakewordlab

If you want a wake word engine that is cheap enough to run continuously on small hardware, wakewordlab is aimed at that use case.

For the local `hey_jarvis` benchmark files used here:

- wakewordlab packaged model (`.wkw`): `244,541` bytes (`~240 KB`)
- openWakeWord pipeline: `3,685,906` bytes total (`~3.5 MB`)
  - mel: `1,087,958`
  - embedding: `1,326,578`
  - head: `1,271,370`

That is roughly a **15x smaller shipped model footprint** for the wakewordlab path tested here.

### Raspberry Pi 3: single-core benchmark

Measured on a Raspberry Pi 3 using one pinned CPU core, single-thread inference, and 30 seconds of 16 kHz synthetic audio:

| Engine | MMAC / invoke | Cadence | MMAC / sec | ms / sec audio | Core-equivalent load | Relative compute |
|---|---:|---:|---:|---:|---:|---:|
| **wakewordlab** (`20260609_142345`, non-streaming ONNX) | `4.81` | `100 ms` | **48.1** | **152.7 ms** | **15.3% of one core** | **1.0x** |
| openWakeWord (`mel + embedding + hey_jarvis head`) | `42.4` | `80 ms` | `530.3` | `405.7 ms` | `40.6% of one core` | `2.66x` |

### Visual comparison
<p align="center">
  <img src="images/Size%20in%20MB%20vs%20Model%20footprint.png" alt="Model footprint" width="32%">
  <img src="images/MMAC_s%20vs%20Raw%20compute%20rate.png" alt="Raw compute rate" width="32%">
  <img src="images/_%20of%20one%20Pi%203%20core%20vs%20Raspberry%20Pi%203%20measured%20CPU%20cost.png" alt="Raspberry Pi 3 measured CPU cost" width="32%">
</p>

### What this means

- **Smaller deployment**: easier to ship, cache, and update on constrained devices
- **Lower raw compute**: about **11x lower MACs per second** than the tested openWakeWord pipeline
- **Lower steady CPU cost**: about **2.7x less CPU time** than the tested openWakeWord pipeline on Pi 3
- **More headroom**: leaves more of the Pi for Home Assistant, audio I/O, automations, and UI work
- **Simple packaging**: a single wakewordlab model file instead of a larger multi-model pipeline

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

## Compute footprint on Home Assistant hardware

Most Home Assistant deployments run on a **Raspberry Pi 4/5** or the dedicated **HA Green** (Rockchip RK3566, quad-core Cortex-A55 @ 1.8 GHz, 4 GB RAM). These are capable machines, but always-on wake word detection still competes with automations, integrations, and add-ons for the same CPU budget.

The table below compares wakewordlab against the two most commonly referenced alternatives: [openWakeWord](https://github.com/dscripka/openWakeWord) (the default HA voice pipeline engine). Numbers are measured from the actual model files.

### Compute comparison

| Engine | MACs / invoke | Cadence | **MMAC/s** | HA Green core load¹ |
|---|---|---|---|---|
| **wakewordlab** (streaming) | ~885 k | 37.5 ms | **~24** | **~1–2%** |
| **wakewordlab** (batch, 100 ms) | ~3.15 M | 100 ms | **~31** | **~2–3%** |
| openWakeWord (embed + 1 head) | ~42 M | 80 ms | **~525** | **~26–53%** |

¹ Cortex-A55 @ 1.8 GHz, single-threaded NEON, ~1–2 GMAC/s per core.

### Why openWakeWord costs so much

openWakeWord is a two-stage system. Stage 1 is a large pretrained Google speech embedding model (42 M MACs) that must run on **every 80 ms audio chunk**, regardless of which or how many wake words you are using. Stage 2 is a tiny per-wake-word head (~50 k MACs) that is nearly free. The embedding is the bottleneck, and it cannot be skipped or shared with any other task.

On a laptop this is invisible. On an HA Green it loads one Cortex-A55 core by **26–53%**, which directly competes with Zigbee polling, recorder writes, and UI requests.

### Why wakewordlab stays cheap

wakewordlab uses a single self-contained convolutional model — no external embedding dependency. The full streaming path costs **~24 MMAC/s**: roughly **22× less than openWakeWord** and comparable to what a Raspberry Pi 3 handles without breaking a sweat.

Additionally, the **Silero VAD pre-filter** (enabled by default) gates inference so the wake word model only runs on frames that contain speech. During silence — which is the majority of time in a home environment — the effective compute drops toward zero.

| Condition | Effective wake word compute |
|---|---|
| Silence (VAD off) | 0 — model not invoked |
| Speech present (VAD on) | ~24 MMAC/s streaming |
| Continuous (VAD disabled) | ~24–31 MMAC/s |

### Each additional wake word

| Engine | Cost of adding one more wake word |
|---|---|
| openWakeWord | +50 k MACs/invoke — nearly free (head only, embed is shared) |
| **wakewordlab** | +~885 k MACs/invoke per model in streaming mode |

openWakeWord has a clear advantage when running many wake words simultaneously. wakewordlab's per-model cost is higher, but because the baseline is ~22× lower, **two wakewordlab models (~48 MMAC/s) still use less than a tenth of the compute of one openWakeWord instance (~525 MMAC/s)**.

### Summary

- **22× lower compute per second** than openWakeWord on the same hardware
- **~1–2% of one CPU core** on HA Green at streaming cadence
- **VAD gating** eliminates nearly all compute during silence
- **Self-contained** — no shared embedding model, no external dependency
- **37.5 ms detection cadence** (faster than openWakeWord's 80 ms)
- Runs comfortably on **Raspberry Pi 3** and all newer Pi and HA hardware

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
