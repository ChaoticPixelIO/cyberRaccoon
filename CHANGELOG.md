# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-05-06

First public release of CyberRaccoon — an AI agent that observes a target
machine via video capture and controls it via emulated HID input.

### Tested configurations

- **Host hardware:** Raspberry Pi 5
- **Capture sources:** AirPlay receiver, HDMI-CSI
- **Executors:** USB HID gadget, Bluetooth HID
- **Target operating systems:** Windows, macOS
  - iPadOS: smoke-tested only

Other Pi models, capture paths, and target OSes may work but are unverified.

### Added

- Plan-then-execute agent loop: task planner, workflow runner, and skill-based
  step execution with pause/resume and budget calibration.
- Web UI with split-pane step inspector, plan discussion/modification, and a
  hardware setup checklist.
- Capture sources: AirPlay receiver and HDMI-CSI (with auto lane detection and
  1080p defaults).
- Executors: combined keyboard+mouse USB HID gadget and Bluetooth HID with
  hardened SDP records.
- One-command installer (`install.sh` / `scripts/setup.sh`) that provisions
  USB gadget, dwc2 overlay, and systemd persistence on the Pi.
- Pluggable LLM backends (Anthropic and OpenAI) with provider-specific settings
  preserved across UI/CLI switches; secrets stored in YAML config.
- Cached target-OS routing, fatal API error gate, operator hints, and
  credential redaction in logs.

### Notes

- Licensed under Apache-2.0.
- On Raspberry Pi, the venv intentionally inherits the system `python3-opencv`
  to preserve GStreamer support required by AirPlay capture.

[Unreleased]: https://github.com/ChaoticPixelIO/cyberRaccoon/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/ChaoticPixelIO/cyberRaccoon/releases/tag/v1.0.0
