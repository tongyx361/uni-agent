from __future__ import annotations

import subprocess
from pathlib import Path, PurePosixPath

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PRIVATE_RUNTIME_TOKEN = "white" + "box"


def _tracked_entries() -> dict[str, str]:
    result = subprocess.run(
        ["git", "ls-files", "--stage", "-z"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
    )
    entries = {}
    for raw_record in result.stdout.split(b"\0"):
        if not raw_record:
            continue
        metadata, raw_path = raw_record.split(b"\t", 1)
        mode, _, stage = metadata.split()
        assert stage == b"0"
        path = raw_path.decode("utf-8", errors="surrogateescape")
        entries[path] = mode.decode("ascii")
    return entries


def test_public_extract_tracked_tree_satisfies_denylist():
    entries = _tracked_entries()
    private_token = _PRIVATE_RUNTIME_TOKEN.encode()

    for path in entries:
        parts = tuple(part.lower() for part in PurePosixPath(path).parts)
        assert ".agents" not in parts
        assert ".cursor" not in parts
        assert _PRIVATE_RUNTIME_TOKEN not in path.lower()
        assert not path.startswith("verl/")
        assert not ("vendor" in parts and "verl" in parts)
        assert not ("third_party" in parts and "verl" in parts)

        source_path = _REPO_ROOT / path
        if path.startswith("uni_agent/") and source_path.suffix == ".py":
            assert private_token not in source_path.read_bytes().lower()

    assert entries.get("verl") == "160000"


def test_public_extract_keeps_public_verl_remote():
    paths = subprocess.run(
        ["git", "config", "-f", ".gitmodules", "--get-regexp", r"^submodule\..*\.path$"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    submodules = [line.split(maxsplit=1) for line in paths.stdout.splitlines()]
    verl_sections = [key.removesuffix(".path") for key, path in submodules if path == "verl"]
    assert len(verl_sections) == 1

    result = subprocess.run(
        ["git", "config", "-f", ".gitmodules", "--get", f"{verl_sections[0]}.url"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "https://github.com/verl-project/verl.git"
