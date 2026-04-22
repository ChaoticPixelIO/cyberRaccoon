"""M3 CLI — manual tool for testing LLM protocols with a screenshot.

Usage::

    python -m cyberraccoon.agent.cli --image screenshot.jpg --goal "Open Notepad"
    python -m cyberraccoon.agent.cli --image screenshot.jpg --goal "Click start menu" --provider openai
    python -m cyberraccoon.agent.cli --image screenshot.jpg --goal "Open Chrome" --protocol prompt
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys

from cyberraccoon.agent.protocols import create_protocol
from cyberraccoon.agent.skills import SkillNotFoundError, load_skills
from cyberraccoon.config import (
    LLM_PROVIDER_DEFAULTS,
    resolve_api_key,
    resolve_provider_base_url,
    resolve_provider_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CyberRaccoon M3 LLM Client CLI"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to screenshot image file (JPEG/PNG)",
    )
    parser.add_argument(
        "--goal",
        required=True,
        help="Task goal description",
    )
    parser.add_argument(
        "--provider",
        default=os.environ.get("CYBERRACCOON_PROVIDER", "openai"),
        help="LLM provider: openai or anthropic (default: openai)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default: from ~/.cyberraccoon/config.yaml, or provider's built-in default)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (default: from ~/.cyberraccoon/config.yaml — set it in the web UI's Config tab)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Custom API base URL (default: from ~/.cyberraccoon/config.yaml)",
    )
    parser.add_argument(
        "--protocol",
        default=os.environ.get("CYBERRACCOON_PROTOCOL", "auto"),
        choices=["auto", "native", "prompt"],
        help="Protocol to use (default: auto)",
    )
    parser.add_argument(
        "--skill", dest="skills", action="append", default=[],
        help="Load application skill(s) (repeatable, e.g. --skill wechat --skill blender)",
    )

    args = parser.parse_args()

    # Resolve provider-scoped LLM defaults now that --provider is known.
    provider_defaults = LLM_PROVIDER_DEFAULTS.get(args.provider, {})
    if args.model is None:
        args.model = (
            resolve_provider_model(args.provider)
            or provider_defaults.get("model", "")
        )
    if args.base_url is None:
        args.base_url = (
            resolve_provider_base_url(args.provider)
            or provider_defaults.get("base_url")
        )

    # Read and encode image
    try:
        with open(args.image, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Error: Image file not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    # Load skills (if specified)
    skill_text: str | None = None
    if args.skills:
        try:
            skill_text = load_skills(args.skills)
        except (SkillNotFoundError, ValueError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    # Create protocol and run one step
    api_key = args.api_key or resolve_api_key(args.provider)
    try:
        protocol = create_protocol(
            provider=args.provider,
            model=args.model,
            api_key=api_key,
            base_url=args.base_url,
            protocol_override=args.protocol,
            skill_text=skill_text,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    result = protocol.step(image_base64, args.goal)

    # Print results
    print("=" * 50)
    print(f"Provider:  {args.provider}")
    print(f"Model:     {args.model}")
    print(f"Protocol:  {args.protocol}")
    print(f"Goal:      {args.goal}")
    print(f"Latency:   {result.latency_ms}ms")
    print(f"Tokens:    {result.input_tokens} in / {result.output_tokens} out")
    print(f"Success:   {result.success}")
    print("=" * 50)

    if result.success:
        if result.is_done:
            print(f"Done:      {result.done_reason}")
        else:
            print(f"Command:   {json.dumps(result.command, ensure_ascii=False)}")
        if result.screen_summary:
            print(f"Summary:   {result.screen_summary}")
    else:
        print(f"Error:     {result.error}")
        if result.raw_text:
            print(f"Raw:       {result.raw_text[:500]}")


if __name__ == "__main__":
    main()
