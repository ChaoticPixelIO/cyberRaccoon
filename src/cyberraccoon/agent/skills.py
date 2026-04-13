"""Skill loader for application-specific LLM instructions.

Skills are directories containing a ``SKILL.md`` file with YAML frontmatter
plus optional resource files. The body of ``SKILL.md`` (frontmatter stripped)
is appended to the system prompt to give the LLM the context it needs to
control uncommon applications (e.g. KiCad, Blender, niche ERP systems).

Layout::

    <skills_dir>/
        <name>/
            SKILL.md         # required: frontmatter (name, description) + body
            <resource>.png   # optional: cheat sheets, screenshots, etc.

``SKILL.md`` format::

    ---
    name: blender
    description: Blender 3D modeling shortcuts and common workflows.
    ---

    # Blender Skill
    ...

Lookup order:
    1. ``~/.cyberraccoon/skills/{name}/SKILL.md``  (user override)
    2. ``<repo>/skills/{name}/SKILL.md``           (bundled)

Usage::

    from cyberraccoon.agent.skills import load_skill, list_skills

    text = load_skill("blender")       # returns SKILL.md body (no frontmatter)
    names = list_skills()              # ["blender", ...]
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml

from cyberraccoon._project_root import PROJECT_ROOT

SKILL_FILENAME = "SKILL.md"


class SkillNotFoundError(FileNotFoundError):
    """Raised when a skill directory cannot be found in any search path."""

    def __init__(self, name: str, searched_paths: list[Path]) -> None:
        self.name = name
        self.searched_paths = searched_paths
        paths_str = ", ".join(str(p) for p in searched_paths)
        super().__init__(
            f"Skill {name!r} not found. Searched: {paths_str}"
        )


class SkillIncompleteError(SkillNotFoundError):
    """Raised when a skill directory exists but contains no ``SKILL.md``."""

    def __init__(self, name: str, skill_dir: Path) -> None:
        self.name = name
        self.skill_dir = skill_dir
        # Use FileNotFoundError init directly to set a custom message rather
        # than the "Searched: ..." format from SkillNotFoundError.
        FileNotFoundError.__init__(
            self,
            f"Skill {name!r} directory {skill_dir} is missing {SKILL_FILENAME}",
        )


class SkillFormatError(ValueError):
    """Raised when ``SKILL.md`` frontmatter is missing or malformed."""


def _bundled_skills_dir() -> Path:
    """Return the path to the bundled skills directory (<repo>/skills/)."""
    return PROJECT_ROOT / "skills"


def _user_skills_dir() -> Path:
    """Return the path to the user skills directory (~/.cyberraccoon/skills/)."""
    return Path.home() / ".cyberraccoon" / "skills"


def _validate_name(name: str) -> None:
    """Validate a skill name, rejecting path traversal and invalid chars."""
    if not name:
        raise ValueError("Skill name must not be empty")
    if "\0" in name:
        raise ValueError("Skill name must not contain null bytes")
    if "/" in name or "\\" in name:
        raise ValueError("Skill name must not contain path separators")
    if ".." in name:
        raise ValueError("Skill name must not contain '..'")


def _resolve_skill_dir(name: str) -> tuple[Path | None, list[Path]]:
    """Find the skill directory in user dir (priority) then bundled dir.

    Returns ``(found_dir_or_None, searched_paths)``.
    """
    searched: list[Path] = []
    for skills_root in (_user_skills_dir(), _bundled_skills_dir()):
        candidate = skills_root / name
        searched.append(candidate)
        if candidate.is_dir():
            return candidate, searched
    return None, searched


def _parse_skill_md(content: str, *, expected_name: str, source_path: Path) -> tuple[dict[str, Any], str]:
    """Parse ``SKILL.md`` content into ``(frontmatter, body)``.

    Validates that frontmatter contains ``name`` (matching *expected_name*) and
    a non-empty ``description``, and that the body is non-empty.
    """
    if not content.startswith("---\n") and not content.startswith("---\r\n"):
        raise SkillFormatError(
            f"{source_path}: must start with YAML frontmatter delimited by '---'"
        )

    # Locate the closing '---' delimiter on its own line.
    after_open = content.split("\n", 1)[1] if content.startswith("---\n") else content.split("\r\n", 1)[1]
    closing = "\n---\n"
    idx = after_open.find(closing)
    if idx == -1:
        # Allow trailing '---' with no terminating newline (end-of-file).
        if after_open.rstrip("\r\n").endswith("\n---") or after_open.rstrip("\r\n") == "---":
            raise SkillFormatError(
                f"{source_path}: frontmatter has no body after closing '---'"
            )
        raise SkillFormatError(
            f"{source_path}: frontmatter is not closed with '---' on its own line"
        )

    frontmatter_text = after_open[:idx]
    body = after_open[idx + len(closing):]
    # Consume the conventional single blank line between '---' and body so
    # that body starts at the first content line (matches Jekyll/Hugo/Pandoc).
    if body.startswith("\n"):
        body = body[1:]

    try:
        data = yaml.safe_load(frontmatter_text) or {}
    except yaml.YAMLError as e:
        raise SkillFormatError(f"{source_path}: invalid YAML frontmatter: {e}") from e

    if not isinstance(data, dict):
        raise SkillFormatError(
            f"{source_path}: frontmatter must be a YAML mapping, got {type(data).__name__}"
        )

    fm_name = data.get("name")
    if not isinstance(fm_name, str) or not fm_name.strip():
        raise SkillFormatError(f"{source_path}: frontmatter requires non-empty 'name'")
    if fm_name != expected_name:
        raise SkillFormatError(
            f"{source_path}: frontmatter name {fm_name!r} does not match skill directory {expected_name!r}"
        )

    description = data.get("description")
    if not isinstance(description, str) or not description.strip():
        raise SkillFormatError(
            f"{source_path}: frontmatter requires non-empty 'description'"
        )

    if not body.strip():
        raise SkillFormatError(f"{source_path}: body is empty after frontmatter")

    return data, body


def _read_skill_md(name: str) -> tuple[dict[str, Any], str, Path]:
    """Locate, read, and parse a skill's ``SKILL.md``.

    Returns ``(frontmatter, body, skill_dir)``.

    Raises:
        SkillNotFoundError: No skill directory found in any search path.
        SkillIncompleteError: Directory exists but ``SKILL.md`` is missing.
        SkillFormatError: ``SKILL.md`` frontmatter is invalid.
        ValueError: File cannot be read.
    """
    _validate_name(name)
    skill_dir, searched = _resolve_skill_dir(name)
    if skill_dir is None:
        raise SkillNotFoundError(name, searched)

    skill_md = skill_dir / SKILL_FILENAME
    if not skill_md.is_file():
        raise SkillIncompleteError(name, skill_dir)

    try:
        content = skill_md.read_text(encoding="utf-8")
    except OSError as e:
        raise ValueError(f"Cannot read skill file {skill_md}: {e}") from e

    frontmatter, body = _parse_skill_md(content, expected_name=name, source_path=skill_md)
    return frontmatter, body, skill_dir


def load_skill(name: str) -> str:
    """Load a skill by name and return its ``SKILL.md`` body (frontmatter stripped).

    Args:
        name: Skill name (the directory name under ``skills/``).

    Returns:
        The markdown body of the skill, with YAML frontmatter removed.

    Raises:
        ValueError: If *name* is invalid.
        SkillNotFoundError: If the skill directory is not found.
        SkillIncompleteError: If the directory has no ``SKILL.md``.
        SkillFormatError: If ``SKILL.md`` frontmatter is malformed.
    """
    _, body, _ = _read_skill_md(name)
    return body


def list_skills() -> list[str]:
    """Return sorted, deduplicated skill directory names from both search paths.

    Includes directories that lack a ``SKILL.md`` so users can see incomplete
    skills (loading one will surface ``SkillIncompleteError``).

    Returns:
        Sorted list of skill names (directory names).
    """
    names: set[str] = set()

    for skills_dir in (_bundled_skills_dir(), _user_skills_dir()):
        if not skills_dir.is_dir():
            continue
        for path in skills_dir.iterdir():
            if path.is_dir() and not path.name.startswith("."):
                names.add(path.name)

    return sorted(names)


def get_skill_source(name: str) -> str:
    """Return the source of a skill: ``"user"`` or ``"bundled"``.

    Resolves directory existence only — does not require ``SKILL.md`` to be
    present or valid.

    Raises:
        ValueError: If *name* is invalid.
        SkillNotFoundError: If no skill directory exists in either search path.
    """
    _validate_name(name)

    user_dir = _user_skills_dir() / name
    if user_dir.is_dir():
        return "user"

    bundled_dir = _bundled_skills_dir() / name
    if bundled_dir.is_dir():
        return "bundled"

    raise SkillNotFoundError(name, [user_dir, bundled_dir])


def get_skill_info(name: str) -> dict[str, str]:
    """Return skill metadata: name, content, source, and description.

    Returns:
        ``{"name": ..., "content": <body>, "source": "user"|"bundled", "description": ...}``

    Raises:
        ValueError: If *name* is invalid.
        SkillNotFoundError: If the skill directory is not found.
        SkillIncompleteError: If the directory has no ``SKILL.md``.
        SkillFormatError: If ``SKILL.md`` frontmatter is malformed.
    """
    frontmatter, body, _ = _read_skill_md(name)
    source = get_skill_source(name)
    return {
        "name": name,
        "content": body,
        "source": source,
        "description": frontmatter["description"],
    }


def save_user_skill(name: str, content: str) -> Path:
    """Write ``SKILL.md`` for a user skill, creating the directory if needed.

    The *content* must be a complete ``SKILL.md`` document including YAML
    frontmatter; the frontmatter ``name`` field must match *name*.

    Args:
        name: Skill name (directory name under ``~/.cyberraccoon/skills/``).
        content: Full ``SKILL.md`` markdown including frontmatter.

    Returns:
        Path to the written ``SKILL.md`` file.

    Raises:
        ValueError: If *name* is invalid or *content* is empty.
        SkillFormatError: If *content* has invalid frontmatter.
    """
    _validate_name(name)
    if not content or not content.strip():
        raise ValueError("Skill content must not be empty")

    # Validate frontmatter before any filesystem writes. The skill_md path
    # passed in is the eventual destination, used purely for error messages.
    user_dir = _user_skills_dir() / name
    skill_md = user_dir / SKILL_FILENAME
    _parse_skill_md(content, expected_name=name, source_path=skill_md)

    user_dir.mkdir(parents=True, exist_ok=True)
    skill_md.write_text(content, encoding="utf-8")
    return skill_md


def load_skills(names: list[str]) -> str | None:
    """Load multiple skills and concatenate their bodies.

    Args:
        names: List of skill names to load.

    Returns:
        Concatenated skill bodies, or ``None`` if the list is empty.

    Raises:
        ValueError: If any name is invalid.
        SkillNotFoundError: If any skill directory is not found.
        SkillIncompleteError: If any skill is missing ``SKILL.md``.
        SkillFormatError: If any ``SKILL.md`` is malformed.
    """
    if not names:
        return None
    texts = [load_skill(name) for name in names]
    return "\n\n".join(texts)


def delete_user_skill(name: str) -> bool:
    """Delete a user skill directory (and any resource files it contains).

    Args:
        name: Skill name (directory name under ``~/.cyberraccoon/skills/``).

    Returns:
        True if the directory was removed, False if it did not exist.

    Raises:
        ValueError: If *name* is invalid.
    """
    _validate_name(name)
    user_dir = _user_skills_dir() / name
    if user_dir.is_dir():
        shutil.rmtree(user_dir)
        return True
    return False
