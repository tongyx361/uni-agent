"""CPU coverage for worker efficiency reporting."""

from __future__ import annotations

import asyncio

import pytest

from uni_agent.framework import entry as entry_module

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def _build_framework():
    from uni_agent.framework.framework import GatewayAgentFramework, _RunnerConfig

    return GatewayAgentFramework(None, runner_registry={"runner": _RunnerConfig("unused", {}, "ray_task", 1)})


@pytest.mark.asyncio
async def test_adapter_combines_session_and_live_model_metrics():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    adapter = entry_module.AgentFrameworkRolloutAdapter()
    adapter.framework_worker = SimpleNamespace(
        get_efficiency_metrics=SimpleNamespace(
            remote=AsyncMock(return_value={"sessions/admitted": 3.0, "admission/wait_s": 12.0})
        )
    )
    adapter.gateway_manager = SimpleNamespace(
        get_efficiency_metrics=AsyncMock(return_value={"model/output_tokens": 100.0, "model/requests_in_flight": 2.0})
    )
    assert await adapter.get_efficiency_metrics() == {
        "sessions/admitted": 3.0,
        "admission/wait_s": 12.0,
        "model/output_tokens": 100.0,
        "model/requests_in_flight": 2.0,
    }


@pytest.mark.asyncio
async def test_session_metrics_exclude_cancelled_capacity_waiters(monkeypatch):
    framework = _build_framework()
    started = asyncio.Event()

    async def episode(**_kwargs):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(framework, "_run_agent_episode", episode)
    kwargs = dict(sample_fields={}, sample_index=0, session_index=0, global_steps=0, sampling_params={})
    running = asyncio.create_task(framework._run_agent_episode_with_concurrency_limit(**kwargs))
    waiting = None
    try:
        await asyncio.wait_for(started.wait(), 2)
        waiting = asyncio.create_task(framework._run_agent_episode_with_concurrency_limit(**kwargs))
        await asyncio.sleep(0)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        metrics = framework.get_efficiency_metrics()
        assert metrics["sessions/admitted"] == metrics["sessions/in_flight"] == 1
        assert metrics["sessions/completed"] == 0
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        metrics = framework.get_efficiency_metrics()
        assert metrics["sessions/admitted"] == metrics["sessions/completed"] == 1
        assert metrics["sessions/in_flight"] == 0
    finally:
        for task in (running, waiting):
            if task is not None:
                task.cancel()
        await asyncio.gather(*(task for task in (running, waiting) if task is not None), return_exceptions=True)
