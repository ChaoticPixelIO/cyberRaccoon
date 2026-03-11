"""Skill loader for application-specific LLM instructions.

Skills are markdown files containing app-specific UI layout, shortcuts,
and workflows that get appended to the system prompt. This gives the
LLM the context it needs to control uncommon applications (e.g. KiCad,
Blender, niche ERP systems).

Lookup order:
    1. ``~/.cyberraccoon/skills/{name}.md``  (user override)
    2. ``<repo>/skills/{name}.md``           (bundled)

Usage::

    from agent.skills import load_skill, list_skills

    text = load_skill("blender")       # returns markdown content
    names = list_skills()              # ["blender", ...]
"""

from __future__ import annotations

from pathlib import Path


class SkillNotFoundError(FileNotFoundError):
    """Raised when a skill file cannot be found in any search path."""

    def __init__(self, name: str, searched_paths: list[Path]) -> None:
        self.name = name
        self.searched_paths = searched_paths
        paths_str = ", ".join(str(p) for p in searched_paths)
        super().__init__(
            f"Skill {name!r} not found. Searched: {paths_str}"
        )


def _bundled_skills_dir() -> Path:
    """Return the path to the bundled skills directory (<repo>/skills/)."""
    return Path(__file__).resolve().parent.parent / "skills"


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


def load_skill(name: str) -> str:
    """Load a skill file by name and return its content.

    Args:
        name: Skill name (without ``.md`` extension).

    Returns:
        The markdown content of the skill file.

    Raises:
        ValueError: If *name* is invalid or the file is empty.
        SkillNotFoundError: If the skill is not found in any search path.
    """
    _validate_name(name)

    searched: list[Path] = []
    filename = f"{name}.md"

    # User dir takes priority
    user_path = _user_skills_dir() / filename
    searched.append(user_path)
    if user_path.is_file():
        try:
            content = user_path.read_text(encoding="utf-8")
        except OSError as e:
            raise ValueError(f"Cannot read skill file {user_path}: {e}") from e
        if not content.strip():
            raise ValueError(f"Skill file is empty: {user_path}")
        return content

    # Bundled fallback
    bundled_path = _bundled_skills_dir() / filename
    searched.append(bundled_path)
    if bundled_path.is_file():
        try:
            content = bundled_path.read_text(encoding="utf-8")
        except OSError as e:
            raise ValueError(f"Cannot read skill file {bundled_path}: {e}") from e
        if not content.strip():
            raise ValueError(f"Skill file is empty: {bundled_path}")
        return content

    raise SkillNotFoundError(name, searched)


def list_skills() -> list[str]:
    """Return sorted, deduplicated skill names from both directories.

    Returns:
        List of skill names (without ``.md`` extension).
    """
    names: set[str] = set()

    for skills_dir in (_bundled_skills_dir(), _user_skills_dir()):
        if skills_dir.is_dir():
            for path in skills_dir.iterdir():
                if path.suffix == ".md" and path.is_file():
                    names.add(path.stem)

    return sorted(names)


def get_skill_source(name: str) -> str:
    """Return the source of a skill: ``"user"``, ``"bundled"``, or raise.

    If the skill exists in the user directory (regardless of whether a
    bundled version also exists), returns ``"user"``.  Otherwise returns
    ``"bundled"``.

    Raises:
        ValueError: If *name* is invalid.
        SkillNotFoundError: If the skill is not found in any search path.
    """
    _validate_name(name)
    filename = f"{name}.md"

    user_path = _user_skills_dir() / filename
    if user_path.is_file():
        return "user"

    bundled_path = _bundled_skills_dir() / filename
    if bundled_path.is_file():
        return "bundled"

    raise SkillNotFoundError(name, [user_path, bundled_path])


def get_skill_info(name: str) -> dict[str, str]:
    """Return skill metadata: name, content, and source.

    Returns:
        ``{"name": ..., "content": ..., "source": "user"|"bundled"}``

    Raises:
        ValueError: If *name* is invalid.
        SkillNotFoundError: If the skill is not found.
    """
    content = load_skill(name)
    source = get_skill_source(name)
    return {"name": name, "content": content, "source": source}


def save_user_skill(name: str, content: str) -> Path:
    """Write a skill file to the user skills directory.

    Creates ``~/.cyberraccoon/skills/`` if it doesn't exist.

    Args:
        name: Skill name (without ``.md`` extension).
        content: Markdown content to write.

    Returns:
        Path to the written file.

    Raises:
        ValueError: If *name* is invalid or *content* is empty.
    """
    _validate_name(name)
    if not content or not content.strip():
        raise ValueError("Skill content must not be empty")

    user_dir = _user_skills_dir()
    user_dir.mkdir(parents=True, exist_ok=True)
    path = user_dir / f"{name}.md"
    path.write_text(content, encoding="utf-8")
    return path


def load_skills(names: list[str]) -> str | None:
    """Load multiple skills and concatenate their content.

    Args:
        names: List of skill names to load.

    Returns:
        Concatenated markdown content, or ``None`` if the list is empty.

    Raises:
        ValueError: If any name is invalid or its file is empty.
        SkillNotFoundError: If any skill is not found.
    """
    if not names:
        return None
    texts = [load_skill(name) for name in names]
    return "\n\n".join(texts)


def delete_user_skill(name: str) -> bool:
    """Delete a skill from the user skills directory.

    Args:
        name: Skill name (without ``.md`` extension).

    Returns:
        True if the file was deleted, False if it didn't exist.

    Raises:
        ValueError: If *name* is invalid.
    """
    _validate_name(name)
    path = _user_skills_dir() / f"{name}.md"
    if path.is_file():
        path.unlink()
        return True
    return False
