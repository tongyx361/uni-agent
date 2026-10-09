"""CPU coverage for adapter completion boundaries and worker efficiency reporting."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from uni_agent.framework import entry as entry_module
from uni_agent.framework.base import AgentFramework

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


@pytest.mark.asyncio
async def test_custom_framework_requires_explicit_admission_support():
    class CustomFramework(AgentFramework):
        @classmethod
        def from_config(cls, **_kwargs):
            return cls()

        async def generate_sequences(self, prompts):
            prompts.append("generated")

    worker_class = entry_module.AgentFrameworkWorker.__ray_metadata__.modified_class
    worker = worker_class.__new__(worker_class)
    worker.framework = CustomFramework()
    prompts = []
    with pytest.raises(AttributeError, match="submit_sessions"):
        await worker.submit_sessions(prompts)
    assert prompts == []  # No fallback to full generation.
    await worker.generate_sequences(prompts)
    assert prompts == ["generated"]


@pytest.mark.parametrize(
    "runner_limits, expected_concurrency",
    [([], 1000), ([4, 6], 1000), ([0, None], 1000), ([600, 700, 0, None], 1300)],
)
def test_adapter_derives_worker_concurrency_from_session_limits(monkeypatch, runner_limits, expected_concurrency):
    worker_class = Mock()
    worker_class.options.return_value = worker_class
    gateway = object()
    monkeypatch.setattr(entry_module, "AgentFrameworkWorker", worker_class)
    monkeypatch.setattr(entry_module, "build_gateway_manager", lambda **_: gateway)
    runners = {
        str(index): {} if limit is None else {"max_concurrent_sessions": limit}
        for index, limit in enumerate(runner_limits)
    }
    config = OmegaConf.create(
        {"actor_rollout_ref": {"rollout": {"n": 8, "custom": {"agent_framework": {"agent_runners": runners}}}}}
    )
    adapter = entry_module.AgentFrameworkRolloutAdapter.create(config=config, llm_client=object())
    worker_class.options.assert_called_once_with(max_concurrency=expected_concurrency)
    assert worker_class.remote.call_args.kwargs["gateway_manager"] is gateway
    assert adapter.framework_worker is worker_class.remote.return_value


@pytest.mark.parametrize(
    "method, rpc, waits",
    [
        ("generate_sequences", "generate_sequences", False),
        ("generate_sequences_and_wait", "generate_sequences", True),
        ("submit_sessions", "submit_sessions", True),
    ],
)
def test_adapter_completion_boundaries(monkeypatch, method, rpc, waits):
    worker = SimpleNamespace(generate_sequences=Mock(), submit_sessions=Mock())
    join = Mock()
    monkeypatch.setattr(entry_module.ray, "get", join)
    adapter = entry_module.AgentFrameworkRolloutAdapter()
    adapter.framework_worker = worker
    prompts = object()
    getattr(adapter, method)(prompts)
    selected, unused = (
        getattr(worker, rpc),
        getattr(worker, "submit_sessions" if rpc == "generate_sequences" else "generate_sequences"),
    )
    selected.remote.assert_called_once_with(prompts)
    unused.remote.assert_not_called()
    if waits:
        join.assert_called_once_with(selected.remote.return_value)
    else:
        join.assert_not_called()


def _build_framework():
    from uni_agent.framework.framework import GatewayAgentFramework, _RunnerConfig

    return GatewayAgentFramework(None, runner_registry={"runner": _RunnerConfig("unused", {}, "ray_task", 1)})


def test_real_ray_snapshot_while_default_generation_slots_are_saturated(tmp_path):
    import ray

    default_slots = 1000  # Ray's default capacity for async actors.

    class BusyWorker(entry_module.AgentFrameworkWorker.__ray_metadata__.modified_class):
        def __init__(self):
            self.framework = _build_framework()
            self.framework._run_agent_episode = self._hold_episode
            self._starts = 0

        async def _hold_episode(self, **_kwargs):
            await asyncio.Event().wait()

        async def generate_sequences(self, marker):
            self._starts += 1
            if self._starts == default_slots:
                Path(marker).touch()
            episode = await self.framework._submit_agent_episode(
                sample_fields={}, sample_index=0, session_index=0, global_steps=0, sampling_params={}
            )
            await episode

    owns_ray = not ray.is_initialized()
    if owns_ray:
        ray.init(address="local", num_cpus=1, num_gpus=0, include_dashboard=False)
    worker = None
    try:
        pythonpath = str(Path(__file__).parent) + os.pathsep + os.environ.get("PYTHONPATH", "")
        # Retain production actor groups and Ray's default generation capacity.
        actor_options = {"num_cpus": 0, **entry_module.AgentFrameworkWorker._default_options}
        worker = (
            ray.remote(**actor_options)(BusyWorker)
            .options(runtime_env={"env_vars": {"PYTHONPATH": pythonpath}})
            .remote()
        )
        marker = tmp_path / "all-generation-slots-started"
        for _ in range(default_slots):
            worker.generate_sequences.remote(str(marker))
        deadline = time.monotonic() + 45
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert marker.exists(), "generation calls did not occupy the default actor slots"
        metrics = ray.get(worker.get_efficiency_metrics.remote(), timeout=10)
        assert metrics["sessions/admitted"] == metrics["sessions/in_flight"] == 1
        assert metrics["sessions/completed"] == 0
    finally:
        if worker is not None:
            ray.kill(worker)
        if owns_ray:
            ray.shutdown()


@pytest.mark.asyncio
async def test_adapter_combines_terminal_task_and_live_model_metrics():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    adapter = entry_module.AgentFrameworkRolloutAdapter()
    adapter.framework_worker = SimpleNamespace(
        get_efficiency_metrics=SimpleNamespace(
            remote=AsyncMock(return_value={"task_reports/count": 3.0, "tool/total_s": 12.0})
        )
    )
    adapter.gateway_manager = SimpleNamespace(
        get_efficiency_metrics=AsyncMock(return_value={"model/output_tokens": 100.0, "model/requests_in_flight": 2.0})
    )
    assert await adapter.get_efficiency_metrics() == {
        "task_reports/count": 3.0,
        "tool/total_s": 12.0,
        "model/output_tokens": 100.0,
        "model/requests_in_flight": 2.0,
    }


async def _efficiency_failing_runner(**kwargs):
    from uni_agent.efficiency import measure_efficiency

    with measure_efficiency("sandbox_startup"):
        await asyncio.sleep(0)
        raise TimeoutError("expected startup failure")


def test_real_ray_runner_reports_to_waiting_worker():
    import os

    import ray

    from uni_agent.framework.framework import _run_agent_runner_ray_task

    class WaitingWorker(entry_module.AgentFrameworkWorker.__ray_metadata__.modified_class):
        def __init__(self):
            self.framework = _build_framework()

        async def run(self):
            try:
                await _run_agent_runner_ray_task.remote(
                    runner_fqn=f"{__name__}._efficiency_failing_runner",
                    runner_kwargs={},
                    raw_prompt=[],
                    session=None,
                    sample_index=0,
                    tools_kwargs=None,
                    log_context=None,
                    efficiency_sink=ray.get_runtime_context().current_actor,
                )
            except ray.exceptions.RayTaskError:
                return self.framework.get_efficiency_metrics()
            raise AssertionError("runner unexpectedly succeeded")

    owns_ray = not ray.is_initialized()
    if owns_ray:
        ray.init(address="local", num_cpus=1, include_dashboard=False)
    worker = None
    try:
        # Pytest imports this directory as a top-level module; let child workers
        # resolve that same module when deserializing the test actor and runner.
        pythonpath = str(Path(__file__).parent) + os.pathsep + os.environ.get("PYTHONPATH", "")
        # The production callback uses the efficiency concurrency group.
        worker = (
            ray.remote(concurrency_groups={"efficiency": 1})(WaitingWorker)
            .options(runtime_env={"env_vars": {"PYTHONPATH": pythonpath}})
            .remote()
        )
        metrics = ray.get(worker.run.remote(), timeout=45)
        assert metrics["task_reports/count"] == 1
        assert metrics["sandbox_startup/error_count"] == 1
        assert metrics["sandbox_startup/count"] == 1
    finally:
        if worker is not None:
            ray.kill(worker)
        if owns_ray:
            ray.shutdown()


@pytest.mark.asyncio
async def test_session_metrics_exclude_cancelled_capacity_waiters(monkeypatch):
    framework = _build_framework()
    started = asyncio.Event()

    async def episode(**_kwargs):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(framework, "_run_agent_episode", episode)
    kwargs = dict(sample_fields={}, sample_index=0, session_index=0, global_steps=0, sampling_params={})
    running = await framework._submit_agent_episode(**kwargs)
    waiting = None
    try:
        await asyncio.wait_for(started.wait(), 2)
        waiting = asyncio.create_task(framework._submit_agent_episode(**kwargs))
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


@pytest.mark.asyncio
async def test_admitted_task_cancelled_before_start_settles_capacity_and_metrics(monkeypatch):
    framework = _build_framework()
    started = []

    async def episode(**_kwargs):
        started.append(True)
        return [], {}

    monkeypatch.setattr(framework, "_run_agent_episode", episode)
    kwargs = dict(sample_fields={}, sample_index=0, session_index=0, global_steps=0, sampling_params={})
    task = await framework._submit_agent_episode(**kwargs)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not started
    metrics = framework.get_efficiency_metrics()
    assert metrics["sessions/admitted"] == metrics["sessions/completed"] == 1
    assert metrics["sessions/in_flight"] == 0

    # The cancelled task returned its admission slot even though its body never ran.
    replacement = await asyncio.wait_for(framework._submit_agent_episode(**kwargs), 2)
    assert await replacement == ([], {})
    metrics = framework.get_efficiency_metrics()
    assert metrics["sessions/admitted"] == metrics["sessions/completed"] == 2
    assert metrics["sessions/in_flight"] == 0
