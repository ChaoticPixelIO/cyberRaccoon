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

- **3 capture sources** — HDMI-CSI bridge (TC358743), AirPlay screen mirroring, or USB HDMI capture card
- **2 HID transports** — Bluetooth HID (wireless) or USB HID Gadget (wired, requires USB power/data splitter on Pi 5)
- **Multiple LLM providers** — OpenAI (GPT), Anthropic (Claude), or any OpenAI-compatible API
- **Input humanization** — Bezier curve mouse movements, variable typing rhythm, jitter, and overshoot to avoid bot detection
- **Web UI + CLI** — Remote task management via browser or interactive terminal REPL
- **Configurable** — CLI flags, environment variables, or YAML config file (3-tier precedence)

## Hardware Requirements

| Component | Purpose | Notes |
|-----------|---------|-------|
| Raspberry Pi 5 | Main controller | Only tested on Pi 5 |
| HDMI-to-CSI module (TC358743) | Screen capture | Optional; uses Pi CSI port, no USB needed |
| USB HDMI capture card | Screen capture | Optional (~$10–20); not fully tested yet |
| USB power/data splitter cable | USB HID output | Optional; needed for USB Gadget mode on Pi 5 |

> **Minimal setup:** Raspberry Pi 5 with Bluetooth + AirPlay — no extra hardware needed. The Pi pairs as a wireless keyboard/mouse via Bluetooth, and captures the screen via AirPlay mirroring. Other capture and input methods require the optional hardware above.

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/ChaoticPixelIO/cyberRaccoon.git
cd cyberRaccoon
pip install -e ".[dev]"
```

### 2. Set your API key

```bash
# OpenAI (default)
export OPENAI_API_KEY="sk-..."

# Anthropic (if using --provider anthropic)
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. Set up hardware (run once on the Pi)

```bash
# Interactive — asks what to configure
sudo scripts/setup.sh

# Or specify components directly
sudo scripts/setup.sh --bt              # Bluetooth HID
sudo scripts/setup.sh --gadget          # USB HID Gadget (needs splitter on Pi 5)
sudo scripts/setup.sh --airplay         # AirPlay capture (optional)
sudo scripts/setup.sh --csi             # CSI HDMI capture (optional)
```

### 4. Run a task

```bash
# Start the Web UI (recommended)
python -m cyberraccoon --web
# Open http://<pi-ip>:8000 in your browser

# Or run a one-shot task from the command line
python -m cyberraccoon --task "Open Notepad and type Hello World" --transport bt

# AirPlay + Bluetooth
python -m cyberraccoon --task "Open Safari" --source airplay --transport bt
```

## Usage

### Web UI (recommended)

```bash
python -m cyberraccoon --web
python -m cyberraccoon --web --host 0.0.0.0 --port 8080
# Open http://<pi-ip>:8000 in your browser
```

The Web UI is the primary way to use CyberRaccoon — it provides live task progress, configuration, and log streaming all in one place.

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
| `--source` | `hdmi` | Capture source: `hdmi`, `csi`, or `airplay` |
| `--device` | `0` | Device index for HDMI/CSI |
| `--rtp-port` | `5004` | RTP port for AirPlay video stream |

**LLM:**

| Flag | Default | Description |
|------|---------|-------------|
| `--provider` | `openai` | LLM provider: `openai` or `anthropic` |
| `--model` | — | Model name (provider default if omitted) |
| `--api-key` | env var | API key (or set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`) |
| `--base-url` | — | Custom API base URL (OpenAI-compatible) |

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
| `--hid-device` | `/dev/hidg0` | HID device path for USB mode (combined keyboard+mouse) |

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

| Source | Status | Best for | Hardware |
|--------|--------|----------|----------|
| CSI (TC358743) | Tested | HDMI input via CSI port (no USB needed) | HDMI-to-CSI bridge module |
| AirPlay | Tested | macOS/iOS wireless mirroring | — (software only: uxplay + GStreamer) |
| HDMI-UVC | Not fully tested | Desktop/laptop with HDMI out | USB HDMI capture card |

> **Note:** The HDMI-UVC (USB capture card) source has not been fully validated on the current Pi 5 setup. CSI and AirPlay are the recommended capture methods.

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

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | **Required** for OpenAI provider |
| `ANTHROPIC_API_KEY` | — | **Required** for Anthropic provider |
| `CYBERRACCOON_PROVIDER` | `openai` | LLM provider (`openai` or `anthropic`) |
| `CYBERRACCOON_MODEL` | — | Model name (provider default if omitted) |
| `CYBERRACCOON_BASE_URL` | — | Custom API base URL |
| `CYBERRACCOON_SOURCE` | `hdmi` | Capture source (`hdmi`, `csi`, `airplay`) |
| `CYBERRACCOON_DEVICE` | `0` | Device index for HDMI/CSI |
| `CYBERRACCOON_TRANSPORT` | `usb` | HID transport (`usb` or `bt`) |
| `CYBERRACCOON_HUMANIZE` | `0` | Enable humanization (`1` to enable) |
| `CYBERRACCOON_WEB_HOST` | `0.0.0.0` | Web server bind address |
| `CYBERRACCOON_WEB_PORT` | `8000` | Web server port |

Config precedence: **environment variables > YAML file > dataclass defaults**

## License

Apache 2.0
