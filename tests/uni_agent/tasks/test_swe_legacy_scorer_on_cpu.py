import hashlib
from pathlib import Path

from uni_agent.tasks.swe_bench import reward_legacy_20261101


def test_legacy_scorer_source_is_the_recorded_experiment_version():
    source = Path(reward_legacy_20261101.__file__).read_bytes()
    assert hashlib.sha256(source).hexdigest() == "fc4ff8cb7374985ce41ce44791165e10023bc71e1db746963b1557ec3f866c14"


def test_legacy_scorer_retains_original_added_test_cleanup(monkeypatch):
    patch = """diff --git a/test_added.py b/test_added.py
new file mode 100644
--- /dev/null
+++ b/test_added.py
@@ -0,0 +1 @@
+assert True
"""
    instance = {"repo": "psf/requests", "version": "2.2", "base_commit": "0" * 40}
    monkeypatch.setattr(reward_legacy_20261101, "get_test_directives", lambda _instance: [])
    old = reward_legacy_20261101._make_eval_script_list(instance, {}, "testbed", "/testbed", instance["base_commit"], patch)
    assert sum(command.startswith("git clean -fx") for command in old) == 1
    assert not any("git rm -rf --ignore-unmatch" in command for command in old)
