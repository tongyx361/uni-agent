"""CPU coverage for the Laminar worker and classic TransferQueue path."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from tests.uni_agent.support import FakeTokenizer
from uni_agent.framework import entry as entry_module
from uni_agent.sandbox.registry import get_sandbox_cls
from verl.utils import tensordict_utils as tu
from verl.workers.rollout.replica import TokenOutput


class _InProcessGatewayManager:
    def __init__(self, actor):
        self._actor = actor
        self.created_session_kwargs = []

    async def create_session(self, session_id: str, **kwargs):
        self.created_session_kwargs.append(dict(kwargs))
        return await self._actor.create_session(session_id, **kwargs)

    async def finalize_session(self, session_id: str):
        return await self._actor.finalize_session(session_id)

    async def abort_session(self, session_id: str) -> None:
        await self._actor.abort_session(session_id)


class _VersionedBackend:
    def __init__(self):
        self.calls = []

    async def generate(
        self,
        session_id,
        *,
        prompt_ids,
        sampling_params,
        image_data=None,
        video_data=None,
        weight_version=None,
    ) -> TokenOutput:
        self.calls.append({"session_id": session_id, "weight_version": weight_version})
        text = r"\boxed{Paris}"
        return TokenOutput(
            token_ids=[ord(char) for char in text],
            log_probs=[-0.1] * len(text),
            stop_reason="completed",
            extra_fields={
                "min_global_steps": weight_version,
                "max_global_steps": weight_version,
            },
        )


def _versioned_prompts(task_config):
    return tu.get_tensordict(
        tensor_dict={
            "raw_prompt": [[{"role": "user", "content": "sample 0"}]],
            "uid": ["uid-0"],
            "data_source": ["local-echo"],
            "reward_model": [{"ground_truth": "ok"}],
            "extra_info": [{"index": 0}],
            "tools_kwargs": [{"task": task_config}],
        },
        non_tensor_dict={
            "global_steps": 7,
        },
    )


def _build_framework(gateway_manager):
    from uni_agent.framework.framework import GatewayAgentFramework

    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "n": 1,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "calculate_log_probs": True,
                    "val_kwargs": {"n": 1, "temperature": 0, "top_p": 1.0, "top_k": -1},
                    "custom": {
                        "agent_framework": {
                            "agent_runners": {
                                "mem_agent": {
                                    "runner_fqn": "uni_agent.framework.task_runner.run_task",
                                    "runner_kwargs": {
                                        "model_name": "test-model",
                                        "report_reward": True,
                                        "task_config_path": str(
                                            Path(__file__).resolve().parents[3]
                                            / "examples/mem_agent/task_config.yaml"
                                        ),
                                    },
                                    "dispatch_mode": "inline_async",
                                }
                            }
                        }
                    },
                }
            }
        }
    )
    return GatewayAgentFramework.from_config(config=config, gateway_manager=gateway_manager)


def test_worker_initializes_transfer_queue(monkeypatch):
    initialized = []
    framework = object()

    class _FakeTQ:
        def init(self):
            initialized.append(True)

    monkeypatch.setattr(entry_module, "tq", _FakeTQ())
    monkeypatch.setattr(entry_module, "init_rollout_trace_config", lambda _config: None)
    monkeypatch.setattr(entry_module, "build_agent_framework", lambda **_: framework)

    worker_class = entry_module.AgentFrameworkWorker.__ray_metadata__.modified_class
    worker = worker_class.__new__(worker_class)
    worker_class.__init__(worker, config=object(), gateway_manager=object())

    assert initialized == [True]
    assert worker.framework is framework


@pytest.mark.asyncio
async def test_worker_and_adapter_forward_framework_metrics():
    expected = {"sink/trajectory_write_failure_count": 2}

    class _Framework:
        def get_metrics(self):
            return dict(expected)

    worker_class = entry_module.AgentFrameworkWorker.__ray_metadata__.modified_class
    worker = worker_class.__new__(worker_class)
    worker.framework = _Framework()
    assert worker.get_metrics() == expected

    class _RemoteMetrics:
        async def remote(self):
            return worker.get_metrics()

    class _FrameworkWorker:
        get_metrics = _RemoteMetrics()

    adapter = entry_module.AgentFrameworkRolloutAdapter()
    adapter.framework_worker = _FrameworkWorker()
    assert await adapter.get_metrics() == expected


def test_adapter_bounds_pending_generation_sessions_and_surfaces_errors(monkeypatch):
    class _GenerationRef:
        def __init__(self) -> None:
            self.error = None

    submitted_refs = []
    ready_refs = set()
    wait_timeouts = []
    version_ref = object()

    def wait(refs, *, num_returns=1, timeout=None):
        wait_timeouts.append(timeout)
        ready = [ref for ref in refs if ref in ready_refs]
        if timeout is None and len(ready) < num_returns:
            ready_refs.add(refs[0])
            ready = [refs[0]]
        selected = ready[:num_returns]
        return selected, [ref for ref in refs if ref not in selected]

    def get(refs):
        if refs is version_ref:
            return 7
        refs = refs if isinstance(refs, list) else [refs]
        for ref in refs:
            if ref.error is not None:
                raise ref.error

    class _LLMClient:
        def wait_until_version_available(self, *, weight_version):
            assert weight_version == 7
            return version_ref

    class _RemoteGeneration:
        def remote(self, _prompts):
            ref = _GenerationRef()
            submitted_refs.append(ref)
            return ref

    class _FrameworkWorker:
        generate_sequences = _RemoteGeneration()

    class _FrameworkWorkerClass:
        @staticmethod
        def remote(**_kwargs):
            return _FrameworkWorker()

    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "n": 8,
                    "custom": {
                        "agent_framework": {
                            "agent_runners": {
                                "primary": {"max_concurrent_sessions": 2048},
                                "bounded": {"max_concurrent_sessions": 1024},
                                "unbounded": {"max_concurrent_sessions": 0},
                            }
                        }
                    },
                }
            }
        }
    )
    monkeypatch.setattr(entry_module.ray, "wait", wait)
    monkeypatch.setattr(entry_module.ray, "get", get)
    monkeypatch.setattr(entry_module, "build_gateway_manager", lambda **_kwargs: object())
    monkeypatch.setattr(entry_module, "AgentFrameworkWorker", _FrameworkWorkerClass)

    adapter = entry_module.AgentFrameworkRolloutAdapter.create(config=config, llm_client=_LLMClient())
    assert adapter._max_inflight_sessions == 1024

    for _ in range(128):
        adapter.generate_sequences(_versioned_prompts({}))
    assert len(submitted_refs) == 128

    adapter.generate_sequences(_versioned_prompts({}))

    assert wait_timeouts[-2:] == [0, None]
    assert len(submitted_refs) == 129
    assert adapter._inflight_sessions == 1024

    submitted_refs[1].error = RuntimeError("generation failed")
    ready_refs.add(submitted_refs[1])
    with pytest.raises(RuntimeError, match="generation failed"):
        adapter.generate_sequences(_versioned_prompts({}))
    assert len(submitted_refs) == 129


def test_adapter_serializes_an_indivisible_batch_larger_than_the_runner_cap(monkeypatch):
    submitted_refs = []
    ready_refs = set()
    wait_timeouts = []

    def wait(refs, *, num_returns=1, timeout=None):
        wait_timeouts.append(timeout)
        ready = [ref for ref in refs if ref in ready_refs]
        if timeout is None and len(ready) < num_returns:
            ready_refs.add(refs[0])
            ready = [refs[0]]
        selected = ready[:num_returns]
        return selected, [ref for ref in refs if ref not in selected]

    class _RemoteGeneration:
        def remote(self, _prompts):
            ref = object()
            submitted_refs.append(ref)
            return ref

    class _FrameworkWorker:
        generate_sequences = _RemoteGeneration()

    class _FrameworkWorkerClass:
        @staticmethod
        def remote(**_kwargs):
            return _FrameworkWorker()

    class _LLMClient:
        def wait_until_version_available(self, *, weight_version):
            return weight_version

    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "n": 8,
                    "custom": {"agent_framework": {"agent_runners": {"task": {"max_concurrent_sessions": 256}}}},
                }
            }
        }
    )
    prompts = tu.get_tensordict(
        tensor_dict={"raw_prompt": [["prompt"]] * 64},
        non_tensor_dict={"global_steps": 7},
    )
    monkeypatch.setattr(entry_module.ray, "wait", wait)
    monkeypatch.setattr(entry_module.ray, "get", lambda ref: ref)
    monkeypatch.setattr(entry_module, "build_gateway_manager", lambda **_kwargs: object())
    monkeypatch.setattr(entry_module, "AgentFrameworkWorker", _FrameworkWorkerClass)

    adapter = entry_module.AgentFrameworkRolloutAdapter.create(config=config, llm_client=_LLMClient())
    adapter.generate_sequences(prompts)
    adapter.generate_sequences(prompts.clone())

    assert len(submitted_refs) == 2
    assert wait_timeouts[-2:] == [0, None]
    assert adapter._inflight_sessions == 256


def test_adapter_preserves_legacy_client_fire_and_forget_semantics(monkeypatch):
    submitted_refs = []

    class _RemoteGeneration:
        def remote(self, _prompts):
            submitted_refs.append(object())
            return submitted_refs[-1]

    class _FrameworkWorker:
        generate_sequences = _RemoteGeneration()

    class _FrameworkWorkerClass:
        @staticmethod
        def remote(**_kwargs):
            return _FrameworkWorker()

    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "n": 8,
                    "custom": {"agent_framework": {"agent_runners": {"task": {"max_concurrent_sessions": 1}}}},
                }
            }
        }
    )
    monkeypatch.setattr(entry_module, "build_gateway_manager", lambda **_kwargs: object())
    monkeypatch.setattr(entry_module, "AgentFrameworkWorker", _FrameworkWorkerClass)

    adapter = entry_module.AgentFrameworkRolloutAdapter.create(config=config, llm_client=object())
    adapter.generate_sequences(_versioned_prompts({}))
    adapter.generate_sequences(_versioned_prompts({}))

    # Legacy clients omit version-availability; runner cap must not enable mailbox admission.
    assert adapter._max_inflight_sessions == 0
    assert len(submitted_refs) == 2


def test_adapter_and_wait_does_not_account_inflight_sessions(monkeypatch):
    submitted_refs = []

    class _RemoteGeneration:
        def remote(self, _prompts):
            submitted_refs.append(object())
            return submitted_refs[-1]

    class _FrameworkWorker:
        generate_sequences = _RemoteGeneration()

    class _FrameworkWorkerClass:
        @staticmethod
        def remote(**_kwargs):
            return _FrameworkWorker()

    class _LLMClient:
        def wait_until_version_available(self, *, weight_version):
            return weight_version

    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "n": 8,
                    "custom": {"agent_framework": {"agent_runners": {"task": {"max_concurrent_sessions": 1}}}},
                }
            }
        }
    )
    monkeypatch.setattr(entry_module.ray, "get", lambda ref: ref)
    monkeypatch.setattr(entry_module, "build_gateway_manager", lambda **_kwargs: object())
    monkeypatch.setattr(entry_module, "AgentFrameworkWorker", _FrameworkWorkerClass)

    adapter = entry_module.AgentFrameworkRolloutAdapter.create(config=config, llm_client=_LLMClient())
    assert adapter._max_inflight_sessions == 1
    adapter.generate_sequences_and_wait(_versioned_prompts({}))
    adapter.generate_sequences_and_wait(_versioned_prompts({}))

    assert len(submitted_refs) == 2
    assert adapter._inflight_sessions == 0
    assert adapter._generation_refs == {}


def test_adapter_waits_for_version_before_remote_submit_and_skips_unversioned_batches(monkeypatch):
    events = []
    wait_ref = object()

    class _LLMClient:
        def wait_until_version_available(self, *, weight_version):
            events.append(("wait", weight_version))
            return wait_ref

    class _RemoteGeneration:
        def remote(self, _prompts):
            events.append("submit")
            return object()

    class _FrameworkWorker:
        generate_sequences = _RemoteGeneration()

    def get(ref):
        assert ref is wait_ref
        events.append("ready")
        return 9

    monkeypatch.setattr(entry_module.ray, "get", get)
    wait_clock = iter([10.0, 12.25, 20.0, 21.0])
    monkeypatch.setattr(entry_module.time, "perf_counter", lambda: next(wait_clock))
    adapter = entry_module.AgentFrameworkRolloutAdapter()
    adapter.framework_worker = _FrameworkWorker()
    adapter.llm_client = _LLMClient()
    adapter._version_availability_waiter = adapter.llm_client.wait_until_version_available

    training_prompts = _versioned_prompts({})
    adapter.generate_sequences(training_prompts)
    assert adapter.get_last_version_availability_wait_duration_s() == 2.25

    validation_prompts = _versioned_prompts({})
    tu.assign_non_tensor_data(validation_prompts, "validate", True)
    adapter.generate_sequences(validation_prompts)
    assert adapter.get_last_version_availability_wait_duration_s() == 1.0
    adapter.generate_sequences(tu.get_tensordict(tensor_dict={"raw_prompt": [["unversioned"]]}))
    assert adapter.get_last_version_availability_wait_duration_s() == 0.0

    assert tu.get(training_prompts, "global_steps") == 9
    assert tu.get(validation_prompts, "global_steps") == 9
    assert events == [
        ("wait", 7),
        "ready",
        "submit",
        ("wait", 7),
        "ready",
        "submit",
        "submit",
    ]


@pytest.mark.asyncio
async def test_local_task_writes_versioned_classic_tq_contract(monkeypatch):
    from uni_agent.framework import framework as framework_module
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.gateway import _GatewayActor

    sandbox_lifecycle = []
    batch_writes = []
    group_writes = []

    local_sandbox_cls = get_sandbox_cls("local")
    original_start = local_sandbox_cls.start
    original_stop = local_sandbox_cls.stop

    async def tracked_start(self):
        sandbox_lifecycle.append("start")
        await original_start(self)

    async def tracked_stop(self):
        sandbox_lifecycle.append("stop")
        await original_stop(self)

    monkeypatch.setattr(local_sandbox_cls, "start", tracked_start)
    monkeypatch.setattr(local_sandbox_cls, "stop", tracked_stop)

    class _FakeTransferQueue:
        async def async_kv_put(self, *, key, partition_id, tag):
            group_writes.append({"key": key, "partition_id": partition_id, "tag": dict(tag)})

        async def async_kv_batch_put(self, *, keys, fields, tags, partition_id):
            batch_writes.append(
                {
                    "keys": list(keys),
                    "partition_id": partition_id,
                    "fields": fields,
                    "tags": [dict(item) for item in tags],
                }
            )

    monkeypatch.setattr(framework_module, "tq", _FakeTransferQueue())

    backend = _VersionedBackend()
    gateway_actor = _GatewayActor(GatewayActorConfig(tokenizer=FakeTokenizer()), backend)
    await gateway_actor.start()
    runtime = _InProcessGatewayManager(gateway_actor)
    task_config = {
        "name": "hotpotqa",
        "prompt": [{"role": "user", "content": "What is the capital of France?"}],
        "ground_truth": ["Paris"],
        "metadata": {"instance_id": "cpu-local", "chunks": []},
    }

    try:
        await _build_framework(runtime).generate_sequences(_versioned_prompts(task_config))
    finally:
        await gateway_actor.shutdown()

    assert sandbox_lifecycle == ["start", "stop"]
    assert runtime.created_session_kwargs[0]["weight_version"] == 7
    assert backend.calls[0]["weight_version"] == 7
    assert batch_writes[0]["keys"] == ["uid-0_0_0"]
    assert batch_writes[0]["tags"][0]["status"] == "success"
    assert tu.get(batch_writes[0]["fields"], "rm_scores")[0][-1].item() == 1.0
    assert group_writes == [
        {
            "key": "uid-0",
            "partition_id": "train",
            "tag": {
                "status": "finished",
                "terminal_schema_version": 1,
                "expected_session_count": 1,
                "successful_session_count": 1,
                "business_failed_session_count": 0,
                "tq_write_failed_session_count": 0,
                "successful_trajectory_count": 1,
            },
        }
    ]
