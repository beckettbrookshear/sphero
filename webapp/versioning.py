from pathlib import Path
from typing import Union

ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = ROOT / ".app_version"
DEFAULT_VERSION = "1.0.0"
IGNORED_DIRECTORIES = {".git", ".venv", "__pycache__", ".pytest_cache", ".learned"}


def _parse_version(version: str) -> tuple[int, int, int]:
    try:
        major, minor, patch = version.strip().split(".")
        return int(major), int(minor), int(patch)
    except (ValueError, AttributeError):
        return 1, 0, 0


def _bump_version(version: str) -> str:
    major, minor, patch = _parse_version(version)
    return f"{major}.{minor}.{patch + 1}"


def _iter_project_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue

        relative_parts = path.relative_to(root).parts
        if any(part in IGNORED_DIRECTORIES for part in relative_parts):
            continue

        if path.name == ".app_version":
            continue

        yield path


def get_app_version(project_root: Union[str, Path, None] = None) -> str:
    root = Path(project_root) if project_root is not None else ROOT
    version_file = root / ".app_version"

    if not version_file.exists():
        version_file.write_text(f"{DEFAULT_VERSION}\n", encoding="utf-8")
        return DEFAULT_VERSION

    current_version = version_file.read_text(encoding="utf-8").strip() or DEFAULT_VERSION
    files = list(_iter_project_files(root))
    if not files:
        return current_version

    newest_source_mtime = max(path.stat().st_mtime for path in files)
    if newest_source_mtime > version_file.stat().st_mtime:
        bumped = _bump_version(current_version)
        version_file.write_text(f"{bumped}\n", encoding="utf-8")
        return bumped

    return current_version
