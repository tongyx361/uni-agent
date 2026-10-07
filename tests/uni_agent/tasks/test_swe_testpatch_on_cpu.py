import shlex
import subprocess
import sys

import pytest

from uni_agent.tasks.swe_bench import reward


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True)


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init")
    git(tmp_path, "config", "user.email", "cpu@example.invalid")
    git(tmp_path, "config", "user.name", "CPU test")
    (tmp_path / "test_modified.py").write_text("def test_existing():\n    assert True\n")
    (tmp_path / "candidate.py").write_text("original = True\n")
    git(tmp_path, "add", "--", "test_modified.py", "candidate.py")
    git(tmp_path, "commit", "-m", "base")
    return tmp_path, git(tmp_path, "rev-parse", "HEAD").strip()


def make_patch(repo):
    (repo / "test_modified.py").write_text("def test_existing():\n    assert 1 == 1\n")
    (repo / "test added.py").write_text("def test_official():\n    assert True\n")
    git(repo, "add", "--", "test_modified.py", "test added.py")
    patch = git(repo, "diff", "--cached")
    git(repo, "reset", "--hard", "HEAD")
    return patch


def run_eval(repo, base, patch, monkeypatch):
    monkeypatch.setitem(
        reward.MAP_REPO_VERSION_TO_SPECS["psf/requests"],
        "cpu",
        {
            "test_cmd": shlex.quote(sys.executable)
            + " -m pytest -rA "
            + shlex.quote("test added.py")
            + " test_modified.py",
        },
    )
    monkeypatch.setattr(reward, "get_test_directives", lambda instance: [])
    commands = reward._make_eval_script_list(
        {"repo": "psf/requests", "version": "cpu", "base_commit": base},
        {},
        "testbed",
        str(repo),
        base,
        patch,
    )
    # Run the actual generated patch/test/reset section; sandbox activation is remote-only.
    start = next(
        i for i, command in enumerate(commands) if command.startswith("git checkout ") or command == "echo 'skip reset'"
    )
    script = "\n".join(["set -uxo pipefail", *commands[start:]])
    return subprocess.run(["bash", "-c", script], cwd=repo, text=True, capture_output=True)


@pytest.mark.parametrize("ignored", [False, True])
def test_added_collision_restores_official_tests_and_preserves_other_candidate_files(repo, monkeypatch, ignored):
    path, base = repo
    patch = make_patch(path)
    (path / "test added.py").write_text('raise AssertionError("candidate collision")\n')
    (path / "test_modified.py").write_text('raise AssertionError("candidate modified test")\n')
    (path / "candidate.py").write_text("candidate_change = True\n")
    (path / "unrelated.txt").write_text("candidate untracked\n")
    if ignored:
        (path / ".gitignore").write_text("test added.py\nunrelated.txt\n")
    before = subprocess.run(["git", "apply", "-"], cwd=path, input=patch, text=True, capture_output=True)
    assert before.returncode != 0
    assert "already exists" in before.stderr

    result = run_eval(path, base, patch, monkeypatch)

    assert result.returncode == 0
    assert "PASSED test added.py::test_official" in result.stdout
    assert "PASSED test_modified.py::test_existing" in result.stdout
    assert reward.START_TEST_OUTPUT in result.stderr and reward.END_TEST_OUTPUT in result.stderr
    assert (path / "test_modified.py").read_text() == "def test_existing():\n    assert True\n"
    assert (path / "candidate.py").read_text() == "candidate_change = True\n"
    assert (path / "unrelated.txt").read_text() == "candidate untracked\n"


def test_apply_failure_stops_before_formal_test_markers(repo, monkeypatch):
    path, base = repo
    patch = make_patch(path).replace("assert True", "assert absent_base_context", 1)
    # Corrupt the modified-file context while retaining a valid unified patch.
    patch = patch.replace("-    assert True", "-    assert missing_context")
    result = run_eval(path, base, patch, monkeypatch)
    assert result.returncode != 0
    assert "patch does not apply" in result.stderr
    assert reward.START_TEST_OUTPUT not in result.stderr
    assert reward.END_TEST_OUTPUT not in result.stderr
    assert "test session starts" not in result.stdout


def test_legitimate_test_failure_keeps_end_marker_and_final_reset(repo, monkeypatch):
    path, base = repo
    patch = make_patch(path).replace("+    assert True", "+    assert False")
    result = run_eval(path, base, patch, monkeypatch)
    assert result.returncode == 0  # Existing shell contract: final test reset succeeds.
    assert "FAILED test added.py::test_official" in result.stdout
    assert reward.START_TEST_OUTPUT in result.stderr and reward.END_TEST_OUTPUT in result.stderr
    assert (path / "test_modified.py").read_text() == "def test_existing():\n    assert True\n"


def test_added_only_patch_cleans_collision_without_modified_reset(repo, monkeypatch):
    path, base = repo
    patch = "".join(str(file) for file in reward.PatchSet(make_patch(path)) if file.is_added_file)
    (path / "test added.py").write_text('raise AssertionError("collision")\n')
    result = run_eval(path, base, patch, monkeypatch)
    assert result.returncode == 0
    assert "PASSED test added.py::test_official" in result.stdout
    assert "PASSED test_modified.py::test_existing" in result.stdout
