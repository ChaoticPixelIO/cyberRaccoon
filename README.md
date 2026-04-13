# CyberRaccoon

AI-powered computer control via Raspberry Pi — capture the screen, let a vision LLM decide what to do, and execute keyboard/mouse input as a hardware device. **Zero software installation on the target machine.** Works with any OS: Windows, macOS, Linux, even BIOS and boot screens.

## How It Works

<p align="center">
  <img src="docs/images/agent-loop.svg" alt="CyberRaccoon agent loop: capture screen, analyze with LLM, execute input" width="800">
</p>

CyberRaccoon runs a synchronous **capture → decide → act** loop:

1. **Capture** the target computer's screen (HDMI capture card, CSI camera, or AirPlay mirroring)
2. **Analyze** the screenshot with a vision LLM, which understands the screen context
3. **Decide** the next action (click, type, scroll, key combo) returned as JSON
4. **Execute** the action as real keyboard/mouse input via USB HID Gadget or Bluetooth HID

The loop repeats until the task is complete. The target computer sees CyberRaccoon as a regular keyboard and mouse — no drivers, agents, or network access required.

## Features

- **4 capture sources** — HDMI-CSI bridge (TC358743), AirPlay screen mirroring, USB HDMI capture card (UVC), or Pi CSI camera (picamera2)
- **2 HID transports** — Bluetooth HID (wireless) or USB HID Gadget (wired, requires USB power/data splitter on Pi 5)
- **Multiple LLM providers** — OpenAI (GPT), Anthropic (Claude), or any OpenAI-compatible API
- **Input humanization** — Bezier curve mouse movements, variable typing rhythm, jitter, and overshoot to avoid bot detection
- **Web UI + CLI** — Remote task management via browser or interactive terminal REPL
- **Configurable** — CLI flags, environment variables, or YAML config file (4-tier precedence: CLI > env > YAML > dataclass defaults)

## Hardware Requirements

| Component | Purpose | Notes |
|-----------|---------|-------|
| Raspberry Pi 5 | Main controller | Only tested on Pi 5 |
| HDMI-to-CSI module (TC358743) | Screen capture | Optional; uses Pi CSI port, no USB needed |
| USB HDMI capture card (UVC) | Screen capture | Optional (~$10–20); not fully tested yet |
| USB power/data splitter cable | USB HID output | Optional; needed for USB Gadget mode on Pi 5 |

> **Minimal setup:** Raspberry Pi 5 with Bluetooth + AirPlay — no extra hardware needed. The Pi pairs as a wireless keyboard/mouse via Bluetooth, and captures the screen via AirPlay mirroring. Other capture and input methods require the optional hardware above.

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/ChaoticPixelIO/cyberRaccoon.git
cd cyberRaccoon
pip install -e ".[dev]"
```

### 2. Set up hardware (run once on the Pi)

```bash
# Interactive — asks what to configure
sudo scripts/setup.sh

# Or specify components directly
sudo scripts/setup.sh --bt              # Bluetooth HID
sudo scripts/setup.sh --gadget          # USB HID Gadget (needs splitter on Pi 5)
sudo scripts/setup.sh --airplay         # AirPlay capture (optional)
sudo scripts/setup.sh --csi             # CSI HDMI capture (optional)
```

### 3. Start the Web UI

```bash
python -m cyberraccoon --web
# Open http://<pi-ip>:8000 in your browser
```

### 4. Configure the LLM

In the **Config** tab, pick a provider and set the API key. The default provider is **OpenAI** (so `OPENAI_API_KEY` works out of the box), but Anthropic is fully supported too — switch providers in the Config tab or set `ANTHROPIC_API_KEY` + `CYBERRACCOON_PROVIDER=anthropic`. Any OpenAI- or Anthropic-compatible model works; the default is `gpt-5.4` but you can override it with `--model` or `CYBERRACCOON_MODEL`.

### 5. Run a task

In the **Task** tab:

1. Pick your **capture source** (CSI, AirPlay, or UVC) and **HID transport** (Bluetooth or USB Gadget).
2. Enter a task (e.g., *"Open Notepad and type Hello World"*) and submit — watch the step-by-step progress.

## Usage

### Web UI (recommended)

The Web UI is the primary way to use CyberRaccoon — it bundles live task progress, configuration, skills management, and log streaming into one place. Quick Start above shows how to launch it on the default port; to bind elsewhere:

```bash
python -m cyberraccoon --web --host 0.0.0.0 --port 8080
```

<p align="center">
  <img src="docs/images/web-ui-task.png" alt="CyberRaccoon Web UI — Task tab with live step progress" width="800">
</p>

**Tabs:**

- **Task** — Submit tasks, view live progress with step-by-step screenshots
- **Config** — Edit capture source, LLM provider, transport settings
- **Skills** — Browse, enable, edit, and create application skills
- **Status** — System overview: module readiness, capture, transport, LLM
- **Debug** — Real-time log streaming via WebSocket with level/module filtering

### Command line

CLI usage requires `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in the environment (the Web UI stores these in config for you).

```bash
# One-shot task
python -m cyberraccoon --task "Click the Start menu"

# Interactive CLI REPL
python -m cyberraccoon --cli

# Web + CLI together
python -m cyberraccoon --web --cli
```

### CLI flags

**Capture:**

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `uvc` | Capture source: `uvc`, `csi`, `airplay`, or `picamera` |
| `--device` | `0` | Device index for UVC/CSI |
| `--rtp-port` | `5004` | RTP port for AirPlay video stream |

**LLM:**

| Flag | Default | Description |
|------|---------|-------------|
| `--provider` | `openai` | LLM provider: `openai` or `anthropic` |
| `--model` | `gpt-5.4` | Any OpenAI- or Anthropic-compatible model (not bound to one) |
| `--api-key` | env var | API key (or set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`) |
| `--base-url` | — | Custom API base URL (OpenAI-compatible) |
| `--protocol` | `auto` | Protocol mode: `auto`, `native` (computer-use tool), or `prompt` (prompt-based) |
| `--no-cache` | off | Disable Anthropic prompt caching |

**Agent:**

| Flag | Default | Description |
|------|---------|-------------|
| `--max-steps` | `50` | Maximum steps per task |
| `--timeout` | `600` | Task timeout in seconds |
| `--delay` | `1.0` | Post-action delay in seconds |

**Executor:**

| Flag | Default | Description |
|------|---------|-------------|
| `--transport` | `usb` | Transport: `usb` or `bt` (Bluetooth) |
| `--hid-device` | `/dev/hidg0` | HID device path for USB mode (combined keyboard+mouse via Report IDs) |
| `--target-os` | `auto` | Target OS for non-ASCII text input via clipboard bridge: `auto`, `windows`, `macos`, `linux` |

**Humanization:**

| Flag | Default | Description |
|------|---------|-------------|
| `--humanize` | off | Enable input humanization |
| `--humanize-preset` | `normal` | Preset: `subtle`, `normal`, or `aggressive` |

**Web server:**

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Web server bind address |
| `--port` | `8000` | Web server port |

**Misc:**

| Flag | Description |
|------|-------------|
| `--verbose` / `-v` | Enable debug logging |

## Capture Sources

| Source (`--source`) | Status | Best for | Hardware |
|---------------------|--------|----------|----------|
| `csi` (TC358743) | Tested | HDMI input via Pi CSI port (no USB needed) | HDMI-to-CSI bridge module |
| `airplay` | Tested | macOS/iOS wireless mirroring | — (software only: uxplay + GStreamer) |
| `uvc` | **WIP** | Desktop/laptop with HDMI out via USB capture | USB HDMI capture card (UVC) |
| `picamera` | **WIP** | Pi CSI camera aimed at a physical screen | Raspberry Pi Camera Module (picamera2) |

> **Note:** `uvc` and `picamera` are still being validated on the current Pi 5 setup. `csi` and `airplay` are the recommended capture methods today.

## Input Humanization

CyberRaccoon can simulate human-like input patterns to avoid bot detection on websites and applications.

- **Mouse:** Bezier curve movements, overshoot-then-correct, micro-tremor before clicks, click jitter
- **Keyboard:** Variable inter-key timing, punctuation pauses, word boundary pauses, speed ramp-up

### Presets

| Preset | Mouse | Keyboard | Use case |
|--------|-------|----------|----------|
| `subtle` | Light curves, minimal jitter | Low timing variance | General automation |
| `normal` | Natural curves, overshoot 10% | Moderate variance, punctuation pauses | Default for most tasks |
| `aggressive` | Slow movements, high jitter, 25% overshoot | Slow typing, high variance | Sites with aggressive bot detection |

```bash
python -m cyberraccoon --task "Fill out the form" --humanize
python -m cyberraccoon --task "Fill out the form" --humanize --humanize-preset aggressive
```

Or via environment variable: `CYBERRACCOON_HUMANIZE=1`

## For Developers

```bash
# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_capture/test_screen_capture.py

# Module CLI tools (test hardware/APIs independently)
python -m cyberraccoon.capture.cli --device 0 --output screenshot.jpg
python -m cyberraccoon.agent.cli --image screenshot.jpg --goal "Open Notepad" --provider anthropic
python -m cyberraccoon.executor.cli click 640 360
python -m cyberraccoon.executor.cli type "hello world"
python -m cyberraccoon.executor.cli key ctrl c
```

### Architecture

Five modules in a hub-and-spoke pattern — M2 (Vision Agent) orchestrates everything:

```
HDMI Input → [M1 Capture] → [M2 Vision Agent] ←→ [M3 LLM Client]
                                    ↓
                             [M4 Executor] → USB/BT HID → Target
                                    ↑
                             [M5 Web UI]
```

## Configuration Reference

Every CLI flag above has a matching `CYBERRACCOON_*` env var — set either. Persistent UI config lives at `~/.cyberraccoon/config.yaml` (override the path with `CYBERRACCOON_CONFIG_PATH`).

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | **Required** for OpenAI provider |
| `ANTHROPIC_API_KEY` | — | **Required** for Anthropic provider |
| `CYBERRACCOON_PROVIDER` | `openai` | LLM provider (`openai` or `anthropic`) |
| `CYBERRACCOON_MODEL` | `gpt-5.4` | Any OpenAI/Anthropic-compatible model ID |
| `CYBERRACCOON_BASE_URL` | — | Custom API base URL |
| `CYBERRACCOON_SOURCE` | `uvc` | Capture source (`uvc`, `csi`, `airplay`, `picamera`) |
| `CYBERRACCOON_DEVICE` | `0` | Device index for UVC/CSI |
| `CYBERRACCOON_TRANSPORT` | `usb` | HID transport (`usb` or `bt`) |
| `CYBERRACCOON_TARGET_OS` | `auto` | Target OS for clipboard bridge (`auto`, `windows`, `macos`, `linux`) |
| `CYBERRACCOON_HUMANIZE` | `0` | Enable humanization (`1` to enable) |
| `CYBERRACCOON_WEB_HOST` | `0.0.0.0` | Web server bind address |
| `CYBERRACCOON_WEB_PORT` | `8000` | Web server port |
| `CYBERRACCOON_CONFIG_PATH` | `~/.cyberraccoon/config.yaml` | YAML config file path |

Config precedence: **CLI flags > environment variables > YAML file > dataclass defaults**

## License

Apache 2.0
