# CyberRaccoon User Guide

This guide covers installation, configuration, and usage of CyberRaccoon in detail. For a quick overview, see the [README](../README.md).

## Table of Contents

- [Installation](#installation)
- [Hardware Setup](#hardware-setup)
  - [USB HID Gadget](#usb-hid-gadget)
  - [Bluetooth HID](#bluetooth-hid)
  - [AirPlay Capture](#airplay-capture)
- [Configuration](#configuration)
  - [API Keys](#api-keys)
  - [Config File](#config-file)
  - [Environment Variables](#environment-variables)
  - [Config Precedence](#config-precedence)
- [Running Tasks](#running-tasks)
  - [One-Shot Mode](#one-shot-mode)
  - [Web UI](#web-ui)
  - [Interactive CLI](#interactive-cli)
- [Capture Sources](#capture-sources)
  - [HDMI](#hdmi)
  - [CSI Camera](#csi-camera)
  - [AirPlay](#airplay)
- [HID Transports](#hid-transports)
  - [USB](#usb)
  - [Bluetooth](#bluetooth)
- [LLM Protocols](#llm-protocols)
- [Skills](#skills)
- [Input Humanization](#input-humanization)
- [Non-ASCII Text Input](#non-ascii-text-input)
- [CLI Reference](#cli-reference)
- [Troubleshooting](#troubleshooting)

---

## Installation

```bash
git clone https://github.com/ChaoticPixelIO/cyberRaccoon.git
cd cyberRaccoon
```

On Raspberry Pi, create a venv with system site-packages (required for OpenCV with GStreamer support):

```bash
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Do not** `pip install opencv-python` inside the venv. It shadows the system OpenCV package and loses GStreamer support, which breaks AirPlay capture. If accidentally installed, fix with:
> ```bash
> pip uninstall opencv-python opencv-python-headless
> ```

---

## Hardware Setup

Each setup script is idempotent and safe to run multiple times. Run them once after a fresh OS install or first boot.

### USB HID Gadget

Creates `/dev/hidg0` (keyboard) and `/dev/hidg1` (mouse) so the Pi appears as a USB keyboard and mouse to the target computer.

```bash
sudo scripts/setup_gadget.sh
```

What it does:

1. Loads the `libcomposite` kernel module
2. Creates a USB Gadget at `/sys/kernel/config/usb_gadget/cyber_raccoon`
3. Configures two HID functions:
   - `hidg0`: Boot keyboard (8-byte reports, US layout)
   - `hidg1`: Absolute-coordinate mouse (7-byte reports, 1280x720 coordinate space)
4. Binds the gadget to the USB Device Controller

Verify the devices exist after running:

```bash
ls -la /dev/hidg*
```

> **Pi 5 note:** The USB-C port on Pi 5 is power-only and cannot be used for USB Gadget. Use Bluetooth HID instead (`--transport bt`), or use a Pi 4B for USB mode.

### Bluetooth HID

Configures the Pi as a Bluetooth keyboard+mouse composite device.

```bash
sudo scripts/setup_bluetooth.sh
```

What it does:

1. Disables the BlueZ input plugin (prevents HID profile conflicts)
2. Sets the Bluetooth device class to keyboard+mouse composite (0x002540)
3. Sets the device name to "CyberRaccoon"
4. Installs a persistent pairing agent (D-Bus service) for automatic "Just Works" pairing
5. Grants Python the required network capabilities via `setcap`

After running, pair from the target computer:

1. Open Bluetooth settings on the target computer
2. Find and pair "CyberRaccoon"
3. Accept the connection — no PIN required

### AirPlay Capture

Installs `uxplay` (an open-source AirPlay receiver) and GStreamer for video decoding.

```bash
sudo scripts/setup_airplay.sh
```

What it does:

1. Installs `uxplay` and GStreamer plugins (base, good, bad, libav)
2. Installs `python3-opencv` with GStreamer support (system package)
3. Installs and enables the Avahi daemon for mDNS discovery
4. Verifies all components are working

On the source Mac/iPhone/iPad:

- **macOS**: System Settings > General > AirDrop & Handoff > AirPlay Receiver, then use Control Center > Screen Mirroring > "CyberRaccoon"
- **iOS/iPadOS**: Control Center > Screen Mirroring > "CyberRaccoon"

---

## Configuration

### API Keys

Set the API key for your LLM provider:

```bash
# Anthropic (default provider)
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenAI (if using --provider openai)
export OPENAI_API_KEY="sk-..."
```

### Config File

CyberRaccoon reads settings from `~/.cyberraccoon/config.yaml`. The file is created automatically when you save settings from the Web UI or CLI REPL.

Example:

```yaml
capture_source: hdmi
executor_transport: bt
target_os: ""                  # auto-detect; or: windows, macos, linux

capture:
  device_index: 0
  target_width: 1280
  target_height: 720
  jpeg_quality: 80

llm:
  provider: anthropic
  model: claude-opus-4-6
  base_url: null
  max_tokens: 1024
  temperature: 0.0

agent:
  max_steps: 50
  max_consecutive_failures: 3
  post_action_delay_s: 1.0
  task_timeout_s: 600.0
  protocol_override: auto      # auto, native, or prompt
  enable_cache: true
  skills:
    - blender

executor:
  keyboard_device: /dev/hidg0
  mouse_device: /dev/hidg1
  screen_width: 1280
  screen_height: 720

network:
  web_host: 0.0.0.0
  web_port: 8000
```

Security notes:

- `wifi_password` is never saved to the config file
- `api_key` is excluded by default
- The config file is created with mode `0o600` (owner read/write only)

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | API key for Anthropic provider |
| `OPENAI_API_KEY` | — | API key for OpenAI provider |
| `CYBERRACCOON_PROVIDER` | `anthropic` | LLM provider |
| `CYBERRACCOON_MODEL` | `claude-opus-4-6` | Model name |
| `CYBERRACCOON_BASE_URL` | — | Custom API base URL |
| `CYBERRACCOON_SOURCE` | `hdmi` | Capture source (`hdmi`, `csi`, `airplay`) |
| `CYBERRACCOON_DEVICE` | `0` | Device index for HDMI/CSI |
| `CYBERRACCOON_TRANSPORT` | `usb` | HID transport (`usb`, `bt`) |
| `CYBERRACCOON_TARGET_OS` | — | Target OS (`windows`, `macos`, `linux`, or empty for auto) |
| `CYBERRACCOON_HUMANIZE` | `0` | Enable humanization (`1` to enable) |
| `CYBERRACCOON_WEB_HOST` | `0.0.0.0` | Web server bind address |
| `CYBERRACCOON_WEB_PORT` | `8000` | Web server port |

### Config Precedence

Settings are merged in this order (highest priority first):

1. **CLI flags** (`--provider openai`, `--max-steps 30`, etc.)
2. **Environment variables** (`CYBERRACCOON_*`)
3. **Config file** (`~/.cyberraccoon/config.yaml`)
4. **Defaults** (hardcoded in dataclasses)

---

## Running Tasks

CyberRaccoon has three operating modes. `--task` runs a single task and exits. `--web` and `--cli` can be used together.

### One-Shot Mode

Run a single task, print results, and exit:

```bash
python main.py --task "Open Notepad and type Hello World"
```

With options:

```bash
# Use OpenAI instead of Anthropic
python main.py --task "Click Start menu" --provider openai --model gpt-4o

# Use CSI camera for capture
python main.py --task "Open Chrome" --source csi

# Use Bluetooth HID
python main.py --task "Open Safari" --transport bt

# AirPlay capture + Bluetooth HID
python main.py --task "Open Firefox" --source airplay --transport bt

# Custom step/timeout limits
python main.py --task "Fill out form" --max-steps 30 --timeout 600

# With humanization
python main.py --task "Login" --humanize --humanize-preset aggressive

# With application skills
python main.py --task "Edit the 3D model" --skill blender
```

Output shows each step as it executes:

```
============================================================
  CyberRaccoon — AI Computer Control
============================================================
  Task:     Open Notepad and type Hello World
  Provider: anthropic / claude-opus-4-6
  Source:   HDMI /dev/video0
  Transport: USB HID
  Limits:   50 steps, 600s timeout
============================================================

Starting task...

  Step  1: click        (46, 695)                  [ok] (1250ms)
  Step  2: type         "notepad"                  [ok] (850ms)
  Step  3: key          enter                      [ok] (120ms)
  Step  4: type         "Hello World"              [ok] (900ms)
  Step  5: done                                    [ok] (80ms)

============================================================
  Result:   OK COMPLETED
  Reason:   Task completed successfully
  Steps:    5
  Tokens:   12840 in / 523 out
  Duration: 14.2s
============================================================
```

### Web UI

Start the web interface:

```bash
python main.py --web
# Open http://<pi-ip>:8000 in your browser
```

The Web UI has five tabs:

**Task** — The main workspace. Connect capture and executor modules, submit tasks, and watch live progress. Each step shows the action, parameters, status, and LLM latency. Click a step to see the full LLM conversation, system prompt, and raw response.

**Config** — Edit LLM, agent, and network settings. Changes are saved to `~/.cyberraccoon/config.yaml`.

**Skills** — Browse, enable, edit, and create application skills. Toggle skills on/off for the current session. Create custom skills that are saved to `~/.cyberraccoon/skills/`.

**Status** — System overview: module readiness, current capture source, transport, LLM provider, task status, and Wi-Fi status.

**Debug** — Real-time log viewer with filtering by level (DEBUG/INFO/WARNING/ERROR) and module (M1-M5). Logs stream via WebSocket.

The Web UI uses REST endpoints for configuration and task control, plus a WebSocket at `/ws` for real-time events (step progress, log messages, status updates).

### Interactive CLI

Start the REPL:

```bash
python main.py --cli

# Or run both Web UI and CLI together
python main.py --web --cli
```

Available commands:

| Command | Description |
|---------|-------------|
| `task run "goal"` | Start a task |
| `task abort` | Abort the running task |
| `task status` | Show current task status |
| `config show` | Display all settings (secrets masked) |
| `config set KEY VALUE` | Update a setting (e.g. `config set llm.model gpt-4o`) |
| `config reset` | Delete config file and revert to defaults |
| `capture test` | Take a test screenshot and show resolution/size |
| `wifi scan` | Scan for available Wi-Fi networks |
| `wifi connect SSID [PASSWORD]` | Connect to a Wi-Fi network |
| `wifi status` | Show Wi-Fi connection status |
| `logs` | Show all logs |
| `logs tail N` | Show last N log lines |
| `logs clear` | Clear log buffer |
| `status` | Show system status |
| `help` | List all commands |
| `quit` / `exit` | Exit the REPL |

Tab completion and command history are available when `prompt_toolkit` is installed.

---

## Capture Sources

### HDMI

Captures the target screen via a USB HDMI capture card using V4L2. This is the most reliable method — it works with any OS and even at BIOS/boot screens.

```bash
python main.py --task "..." --source hdmi --device 0
```

Requirements:

- USB HDMI capture card (generic UVC cards work, ~$10-20)
- HDMI cable from target computer to capture card
- Capture card plugged into Pi USB port

Test capture independently:

```bash
python -m capture.cli --device 0 --output screenshot.jpg
```

### CSI Camera

Uses a Raspberry Pi camera module (CSI connector) pointed at the target screen. Useful when HDCP blocks HDMI capture, or when no capture card is available.

```bash
python main.py --task "..." --source csi
```

Requirements:

- Raspberry Pi camera module connected to CSI port
- `picamera2` library (included in Raspberry Pi OS)
- Camera enabled in `raspi-config`

### AirPlay

Receives screen mirroring from macOS/iOS devices over the network. No cables or capture hardware needed.

```bash
python main.py --task "..." --source airplay
```

Requirements:

- `uxplay` and GStreamer installed (`sudo scripts/setup_airplay.sh`)
- Pi and source device on the same network
- Source device mirroring to "CyberRaccoon"

Two capture modes are used automatically:

- **RTP mode** (uxplay >= 1.73): Decodes video frames in real-time via a GStreamer pipeline
- **File mode** (older uxplay): Reads JPEG files written to disk by uxplay

---

## HID Transports

### USB

The Pi appears as a USB keyboard and mouse via USB HID Gadget. Requires a physical USB connection (OTG cable on Pi 4B).

```bash
python main.py --task "..." --transport usb
```

- Keyboard: `/dev/hidg0` (8-byte boot keyboard reports)
- Mouse: `/dev/hidg1` (7-byte absolute-coordinate reports, 1280x720 space)
- Requires `sudo` or membership in the `hidg_users` group

> Not available on Pi 5 (USB-C is power-only). Use Bluetooth instead.

### Bluetooth

The Pi pairs as a wireless Bluetooth keyboard and mouse. Works with Pi 4B and Pi 5.

```bash
python main.py --task "..." --transport bt
```

Connection flow:

1. Run `sudo scripts/setup_bluetooth.sh` (once)
2. Pair "CyberRaccoon" from the target computer's Bluetooth settings
3. Start a task with `--transport bt`
4. The executor waits for the host to connect (60s timeout), then sends HID reports over Bluetooth L2CAP

---

## LLM Protocols

CyberRaccoon supports three protocol modes for communicating with the LLM. The default (`auto`) picks the best one based on provider and model.

| Mode | Flag | Description |
|------|------|-------------|
| Auto | `--protocol auto` | Selects native for supported models, prompt-based otherwise |
| Native | `--protocol native` | Uses the provider's structured computer-use API (Anthropic `computer_20251124` tool or OpenAI Responses API) |
| Prompt | `--protocol prompt` | LLM returns JSON in free text, parsed via 3-level fallback (direct parse, markdown extraction, regex) |

```bash
# Force prompt-based protocol (works with any model)
python main.py --task "..." --protocol prompt
```

Prompt caching (Anthropic) is enabled by default. Disable with `--no-cache`.

---

## Skills

Skills are markdown files that give the LLM app-specific context — UI layouts, keyboard shortcuts, and workflows for specific applications. They are appended to the system prompt.

### Using skills

```bash
# Single skill
python main.py --task "Edit the 3D model" --skill blender
```

Or in the config file:

```yaml
agent:
  skills:
    - blender
```

### Skill lookup order

1. **User skills**: `~/.cyberraccoon/skills/{name}.md` (highest priority)
2. **Bundled skills**: `<repo>/skills/{name}.md`

User skills override bundled ones with the same name.

### Creating custom skills

Create a markdown file describing the application:

```markdown
# My Application

## Window Layout
- Top menu bar with File, Edit, View menus
- Left sidebar with navigation tree
- Main content area in the center
- Status bar at the bottom

## Keyboard Shortcuts
- Ctrl+N: New document
- Ctrl+S: Save
- Ctrl+Z: Undo

## Common Workflows

### Creating a new project
1. Click File > New Project
2. Enter project name in the dialog
3. Click "Create"
```

Save it to `~/.cyberraccoon/skills/myapp.md`, then use `--skill myapp`.

You can also create and edit skills from the Web UI's Skills tab.

### Listing available skills

```bash
# In the CLI REPL
> status   # shows active skills

# Via the Web UI Skills tab
# Or programmatically:
python -c "from agent.skills import list_skills; print(list_skills())"
```

---

## Input Humanization

Humanization simulates human-like input patterns to avoid bot detection. When enabled, mouse movements follow Bezier curves with timing variance and jitter, and typing uses variable inter-key delays with punctuation pauses.

### Enabling

```bash
# Via CLI flag
python main.py --task "..." --humanize

# With a preset
python main.py --task "..." --humanize --humanize-preset aggressive

# Via environment variable
export CYBERRACCOON_HUMANIZE=1
python main.py --task "..."
```

### Presets

| Preset | Mouse behavior | Keyboard behavior | Best for |
|--------|---------------|-------------------|----------|
| `subtle` | Light curves, 1px jitter, 5% overshoot | Low timing variance | General automation |
| `normal` | Natural curves, 2px jitter, 10% overshoot, hand tremor | Moderate variance, punctuation pauses | Most tasks |
| `aggressive` | Slow movements, 4px jitter, 25% overshoot, exaggerated curves | Slow typing, high variance | Aggressive bot detection |

### Fine-tuning

For full control, set individual parameters via environment variables:

```bash
CYBERRACCOON_HUMANIZE=1
CYBERRACCOON_HUMANIZE_SPEED=2.0          # mouse speed multiplier (>1 = slower)
CYBERRACCOON_HUMANIZE_NOISE=1.5          # Bezier curve noise (0 = straight)
CYBERRACCOON_HUMANIZE_JITTER=4           # click jitter in pixels
CYBERRACCOON_HUMANIZE_VARIANCE=0.5       # per-delay variance [0-1]
CYBERRACCOON_HUMANIZE_OVERSHOOT_PROB=0.2 # overshoot probability [0-1]
CYBERRACCOON_HUMANIZE_OVERSHOOT_PX=25    # max overshoot distance in pixels
CYBERRACCOON_HUMANIZE_MICRO_ENABLED=1    # hand tremor before click (0/1)
CYBERRACCOON_HUMANIZE_MICRO_AMP=1.5      # tremor amplitude in pixels
CYBERRACCOON_HUMANIZE_TYPING_SPEED=1.5   # typing speed multiplier (>1 = slower)
CYBERRACCOON_HUMANIZE_TYPING_VARIANCE=0.5
CYBERRACCOON_HUMANIZE_PUNCT_PAUSE=120    # extra ms after punctuation
CYBERRACCOON_HUMANIZE_WORD_PAUSE=60      # extra ms after spaces
```

---

## Non-ASCII Text Input

USB HID keyboards only support US keyboard scancodes. Characters like Chinese, Japanese, Korean, emoji, or accented letters cannot be typed directly. CyberRaccoon handles this automatically via a clipboard bridge.

When the text to type contains non-ASCII characters:

1. ASCII segments are typed normally via HID
2. Non-ASCII segments are copied to the target's clipboard and pasted

The paste method depends on the target OS:

| Target OS | Clipboard command | Paste shortcut |
|-----------|------------------|----------------|
| Windows | PowerShell `Set-Clipboard` | Ctrl+V |
| macOS | `pbcopy` | Cmd+V |
| Linux | `xclip` | Ctrl+V |

Set the target OS explicitly if auto-detection doesn't work:

```bash
python main.py --task "..." --target-os windows
```

Or in the config file:

```yaml
target_os: macos
```

---

## CLI Reference

### Main program

```
python main.py [MODE] [OPTIONS]

Modes (at least one required):
  --task GOAL        Run a single task and exit
  --web              Start the Web UI server
  --cli              Start the interactive CLI REPL

Capture:
  --source {hdmi,csi,airplay}   Capture source (default: hdmi)
  --device N                    Device index (default: 0)
  --rtp-port N                  RTP port for AirPlay (default: 5004)

LLM:
  --provider NAME               anthropic or openai (default: anthropic)
  --model NAME                  Model name (default: claude-opus-4-6)
  --api-key KEY                 API key (default: from env var)
  --base-url URL                Custom API base URL
  --protocol {auto,native,prompt}  Protocol mode (default: auto)
  --no-cache                    Disable Anthropic prompt caching

Agent:
  --max-steps N                 Max steps per task (default: 50)
  --timeout SECONDS             Task timeout (default: 600)
  --delay SECONDS               Post-action delay (default: 1.0)

Executor:
  --transport {usb,bt}          USB HID or Bluetooth HID (default: usb)
  --keyboard PATH               Keyboard device (default: /dev/hidg0)
  --mouse PATH                  Mouse device (default: /dev/hidg1)
  --target-os {auto,windows,macos,linux}  Target OS (default: auto)

Humanization:
  --humanize                    Enable input humanization
  --humanize-preset {subtle,normal,aggressive}  Preset (default: normal)

Skills:
  --skill NAME                  Load a skill (repeatable)

Web server:
  --host ADDRESS                Bind address (default: 0.0.0.0)
  --port N                      Port (default: 8000)

Misc:
  --verbose, -v                 Enable debug logging
```

### Module CLI tools

Test individual modules independently:

```bash
# Capture a screenshot
python -m capture.cli --device 0 --output screenshot.jpg
python -m capture.cli --source csi --output camera.jpg
python -m capture.cli --source airplay --output airplay.jpg

# Test LLM with a screenshot
python -m agent.cli --image screenshot.jpg --goal "Open Notepad"
python -m agent.cli --image screenshot.jpg --goal "Click Start" --provider openai

# Execute HID commands directly
python -m executor.cli click 640 360
python -m executor.cli double_click 640 360
python -m executor.cli type "hello world"
python -m executor.cli key ctrl c
python -m executor.cli key alt f4
python -m executor.cli scroll down
python -m executor.cli drag 100 200 400 500
```

---

## Troubleshooting

### Capture

**"/dev/video0: No such file or directory"**
- HDMI capture card not connected or not recognized. Check `ls /dev/video*`.
- Try a different device index: `--device 1`.

**Black or blank screenshots**
- Ensure the target computer has video output on the HDMI port.
- Some capture cards take a few seconds to sync after connection.
- HDCP-protected content may appear black — try a CSI camera instead.

**AirPlay: "GStreamer not available"**
- Run `sudo scripts/setup_airplay.sh`.
- Ensure venv was created with `--system-site-packages`.
- If `opencv-python` was pip-installed, uninstall it: `pip uninstall opencv-python opencv-python-headless`.

**AirPlay: device not visible**
- Ensure Avahi is running: `sudo systemctl status avahi-daemon`.
- Pi and source device must be on the same network.

### Executor

**"Permission denied: /dev/hidg0"**
- Run with `sudo`, or add your user to the `hidg_users` group.

**"Failed to open executor" (Bluetooth)**
- Run `sudo scripts/setup_bluetooth.sh`.
- Ensure Bluetooth is enabled: `sudo bluetoothctl power on`.
- Pair "CyberRaccoon" from the target computer first.

**USB HID not working on Pi 5**
- Pi 5's USB-C port is power-only. Use `--transport bt` instead.

### LLM

**"API key not set"**
- Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your environment.
- On the Pi, API keys are typically in `~/.apikeys` (sourced by `~/.bashrc`). Use `source ~/.apikeys` before running.

**"Model not found" or 404 errors**
- Check the model name matches your provider. Anthropic models start with `claude-`, OpenAI models with `gpt-`.
- If using a custom base URL, verify it's correct.

### General

**Task completes too quickly without doing anything useful**
- Increase `--max-steps` and `--timeout`.
- Add `--delay 2.0` for slower applications that need time to render.

**Task keeps failing / consecutive failure limit reached**
- Check that the capture source is producing valid screenshots (`python -m capture.cli ...`).
- Check that the executor can send input (`python -m executor.cli click 640 360`).
- Try a different protocol: `--protocol prompt` for broader model compatibility.

**Web UI not accessible from another device**
- Ensure `--host 0.0.0.0` is set (not `127.0.0.1`).
- Check the Pi's firewall allows the port (default 8000).
- Kill any existing server first: `ps aux | grep "python.*main.py"` and kill the PID.
