"""CPU checks for ``UNI_AGENT_SANDBOX_PLUGINS`` overlay discovery."""

from __future__ import annotations

import pytest

from uni_agent.sandbox.registry import (
    SANDBOX_MODULES,
    SANDBOX_PLUGIN_MODULES_ENV,
    SANDBOX_REGISTRY,
    get_sandbox_cls,
)

_PROBE_MODULE = "tests.uni_agent.sandbox.plugin_probe_provider"
_PROBE_NAME = "plugin_probe"


def _unload_probe() -> None:
    SANDBOX_REGISTRY.pop(_PROBE_NAME, None)


@pytest.fixture(autouse=True)
def _isolate_probe_registry():
    _unload_probe()
    yield
    _unload_probe()


def test_in_tree_map_does_not_name_overlay_providers():
    assert "seed" not in SANDBOX_MODULES
    assert _PROBE_NAME not in SANDBOX_MODULES


def test_env_plugin_registers_provider(monkeypatch):
    monkeypatch.setenv(SANDBOX_PLUGIN_MODULES_ENV, _PROBE_MODULE)
    cls = get_sandbox_cls(_PROBE_NAME)
    assert cls.__name__ == "PluginProbeSandbox"
    assert cls.__module__ == _PROBE_MODULE


def test_missing_plugin_module_fails_closed(monkeypatch):
    monkeypatch.setenv(SANDBOX_PLUGIN_MODULES_ENV, "not_a_real_sandbox_plugin_module")
    with pytest.raises(ImportError, match="not_a_real_sandbox_plugin_module"):
        get_sandbox_cls(_PROBE_NAME)


def test_unknown_provider_stays_fail_closed(monkeypatch):
    monkeypatch.delenv(SANDBOX_PLUGIN_MODULES_ENV, raising=False)
    with pytest.raises(ValueError, match="Unknown sandbox provider"):
        get_sandbox_cls("not-a-real-provider")
