"""Factory entry + trainer-facing adapter for the agent framework stack.

`build_gateway_manager` owns gateway-universal wiring (driver-side); the trainer
adapter creates the manager and injects it so the framework only handles its own
agent runner, reward dispatch, and framework-specific config fields.

`AgentFrameworkRolloutAdapter` satisfies the trainer's
`agent_loop_manager_class` extension-point contract; recipes wire it in via
yaml without authoring per-recipe glue:

    actor_rollout_ref.rollout.agent.agent_loop_manager_class:
        uni_agent.framework.entry.AgentFrameworkRolloutAdapter
"""

from __future__ import annotations

import time

import ray
from omegaconf import OmegaConf
from tensordict import TensorDict

from uni_agent.framework.base import AgentFramework
from uni_agent.gateway.config import GatewayActorConfig
from uni_agent.gateway.manager import GatewayManager
from uni_agent.rlinsight_adapter import init_rollout_trace_config
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.transferqueue_utils import tq
from verl.workers.config.model import HFModelConfig

_DEFAULT_FRAMEWORK_CLASS = "uni_agent.framework.framework.GatewayAgentFramework"


def build_gateway_manager(*, config, llm_client) -> GatewayManager:
    """Spawn the gateway actor pool (driver-side, driver-owned) and return its manager."""
    # TODO(phase-b): switch this to actor_rollout_ref.rollout.agent_framework.*
    af_cfg = OmegaConf.select(config, "actor_rollout_ref.rollout.custom.agent_framework", default={}) or {}
    apply_chat_template_kwargs = OmegaConf.select(config, "data.apply_chat_template_kwargs", default={}) or {}
    if OmegaConf.is_config(apply_chat_template_kwargs):
        apply_chat_template_kwargs = OmegaConf.to_container(apply_chat_template_kwargs, resolve=True)

    # Match AgentLoopWorker pattern: self-load tokenizer/processor via HFModelConfig.
    model_config: HFModelConfig = omega_conf_to_dataclass(config.actor_rollout_ref.model)
    # TODO: Gateway session capacity is prompt_length + response_length, which can
    # disagree with actor_rollout_ref.rollout.max_model_len (engine context window
    # or extra slack). Forward that knob into GatewayActorConfig once the override
    # path is restored so session clipping matches the engine budget.
    gateway_actor_config = GatewayActorConfig(
        tokenizer=model_config.tokenizer,
        processor=model_config.processor,
        tool_parser_name=config.actor_rollout_ref.rollout.get("multi_turn", {}).get("format"),
        rollout_backend=config.actor_rollout_ref.rollout.get("name"),
        enable_tool_parser_cache=af_cfg.get("enable_tool_parser_cache", True),
        apply_chat_template_kwargs=dict(apply_chat_template_kwargs),
        prompt_length=config.actor_rollout_ref.rollout.prompt_length,
        response_length=config.actor_rollout_ref.rollout.response_length,
        enable_last_assistant_rollback=af_cfg.get("enable_last_assistant_rollback", True),
    )

    return GatewayManager(
        llm_client=llm_client,
        gateway_count=int(af_cfg["gateway_count"]),
        gateway_actor_config=gateway_actor_config,
    )


def build_agent_framework(
    *,
    config,
    gateway_manager,
    reward_loop_worker_handles=None,
) -> AgentFramework:
    """Wire the configured framework subclass over an injected gateway manager."""
    # TODO(phase-b): switch this to actor_rollout_ref.rollout.agent_framework.*
    af_cfg = OmegaConf.select(config, "actor_rollout_ref.rollout.custom.agent_framework", default={}) or {}
    model_config: HFModelConfig = omega_conf_to_dataclass(config.actor_rollout_ref.model)

    framework_cls = load_class_from_fqn(str(af_cfg.get("framework_class_fqn", _DEFAULT_FRAMEWORK_CLASS)))
    return framework_cls.from_config(
        config=config,
        gateway_manager=gateway_manager,
        processor=model_config.processor,
        reward_loop_worker_handles=reward_loop_worker_handles,
    )


@ray.remote
class AgentFrameworkWorker:
    """Ray actor host: initializes TQ in this process and owns one AgentFramework.

    Construction is synchronous (no async setup round-trip). Ray serializes the
    injected manager into a worker-side copy that owns live session mappings;
    the original driver manager remains the gateway actor-pool owner.
    """

    def __init__(self, *, config, gateway_manager, reward_loop_worker_handles=None) -> None:
        tq.init()
        init_rollout_trace_config(config)
        self.framework = build_agent_framework(
            config=config,
            gateway_manager=gateway_manager,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

    async def generate_sequences(self, prompts) -> None:
        # Run one already-submitted batch. Session occupancy is the framework
        # semaphore; Ray mailbox admission belongs in the adapter, before ``.remote()``.
        await self.framework.generate_sequences(prompts)

    def get_metrics(self) -> dict[str, int | float]:
        return self.framework.get_metrics()


class AgentFrameworkRolloutAdapter:
    """Trainer-facing adapter satisfying the `agent_loop_manager_class` contract.

    Holds zero recipe-specific logic; every agent-framework recipe wires the
    same class in yaml. The adapter owns the gateway manager (driver-side) and
    injects it into the framework worker. Laminar mailbox admission stays in
    :meth:`generate_sequences`; this is the single production manager class.
    """

    def __init__(self) -> None:
        self.framework_worker = None
        # Driver-owned so the gateway actors outlive the framework worker; also
        # the handle through which teardown can be driven once a call site exists.
        self.gateway_manager = None
        self.llm_client = None
        self._version_availability_waiter = None
        self._max_inflight_sessions = 0
        self._num_samples_per_prompt = 1
        self._num_val_samples_per_prompt = 1
        self._generation_refs: dict[ray.ObjectRef, int] = {}
        self._inflight_sessions = 0
        self._last_version_availability_wait_duration_s = 0.0

    @classmethod
    def create(
        cls,
        *,
        config,
        llm_client,
        teacher_client=None,
        reward_loop_worker_handles=None,
        **_,
    ) -> AgentFrameworkRolloutAdapter:
        if teacher_client is not None:
            raise ValueError(
                "AgentFrameworkRolloutAdapter does not support teacher_client yet; "
                "disable teacher policy/distillation or use an AgentLoopManager that supports it."
            )

        gateway_manager = build_gateway_manager(config=config, llm_client=llm_client)
        framework_worker = AgentFrameworkWorker.remote(
            config=config,
            gateway_manager=gateway_manager,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

        instance = cls()
        instance.framework_worker = framework_worker
        instance.gateway_manager = gateway_manager
        instance.llm_client = llm_client
        # Laminar clients expose version pinning. Unbounded fire-and-forget
        # submit can then park stale batches in the worker mailbox, so inflight
        # is bounded to the runner cap only when that capability exists. Legacy
        # PPO / inference clients omit it; their callers are demand-driven or join
        # the batch, and keep unbounded submit.
        version_availability_waiter = getattr(llm_client, "wait_until_version_available", None)
        if callable(version_availability_waiter):
            instance._version_availability_waiter = version_availability_waiter
        af_cfg = OmegaConf.select(config, "actor_rollout_ref.rollout.custom.agent_framework", default={}) or {}
        runner_caps = [
            int(runner_config.get("max_concurrent_sessions", 0) or 0)
            for runner_config in af_cfg.get("agent_runners", {}).values()
        ]
        if instance._version_availability_waiter is not None:
            instance._max_inflight_sessions = min((cap for cap in runner_caps if cap > 0), default=0)
        instance._num_samples_per_prompt = int(config.actor_rollout_ref.rollout.get("n", 1))
        val_kwargs = config.actor_rollout_ref.rollout.get("val_kwargs", {}) or {}
        instance._num_val_samples_per_prompt = int(val_kwargs.get("n", instance._num_samples_per_prompt))
        return instance

    def generate_sequences(self, prompts) -> None:
        """Submit a TQ batch generation task without waiting for rollout results.

        Laminar Dispatcher calls this on a timer and does not join the worker.
        Mailbox admission therefore lives here, immediately before ``.remote()``:
        the worker cannot refuse an already-queued Ray task, and Dispatcher does
        not hold generation refs or runner caps. Do not split this into a public
        admit/query API or a Laminar subclass; classic clients keep fire-and-forget.
        """
        if self.framework_worker is None:
            raise RuntimeError("framework must be initialized before generate_sequences")

        is_validate = bool(tu.get(prompts, "validate", False))
        num_samples_per_prompt = self._num_val_samples_per_prompt if is_validate else self._num_samples_per_prompt
        batch_sessions = len(prompts) * num_samples_per_prompt
        accounted_sessions = self._wait_for_inflight_capacity(batch_sessions)
        self._wait_until_version_available(prompts)
        generation_ref = self.framework_worker.generate_sequences.remote(prompts)
        if accounted_sessions > 0:
            self._generation_refs[generation_ref] = accounted_sessions
            self._inflight_sessions += accounted_sessions
        return None

    def _wait_for_inflight_capacity(self, batch_sessions: int) -> int:
        """Block until this batch can be submitted without exceeding the runner cap.

        Returns the session count to register after submit. Zero means admission
        is off (legacy client) and the caller must not track the generation ref.
        Must be followed immediately by version wait + submit in
        :meth:`generate_sequences`.
        """
        if self._max_inflight_sessions <= 0:
            return 0

        accounted_sessions = min(batch_sessions, self._max_inflight_sessions)
        if self._generation_refs:
            ready_refs, _ = ray.wait(
                list(self._generation_refs),
                num_returns=len(self._generation_refs),
                timeout=0,
            )
            if ready_refs:
                self._inflight_sessions -= sum(self._generation_refs.pop(ref) for ref in ready_refs)
                ray.get(ready_refs)

        while self._inflight_sessions + accounted_sessions > self._max_inflight_sessions:
            ready_refs, _ = ray.wait(list(self._generation_refs), num_returns=1)
            ready_ref = ready_refs[0]
            self._inflight_sessions -= self._generation_refs.pop(ready_ref)
            ray.get(ready_ref)
        return accounted_sessions

    def _wait_until_version_available(self, prompts: TensorDict) -> None:
        self._last_version_availability_wait_duration_s = 0.0
        weight_version = tu.get(prompts, "global_steps")
        if weight_version is None:
            return
        if self._version_availability_waiter is None:
            return
        wait_started_at = time.perf_counter()
        try:
            available_weight_version = ray.get(self._version_availability_waiter(weight_version=weight_version))
        finally:
            self._last_version_availability_wait_duration_s = time.perf_counter() - wait_started_at
        if type(available_weight_version) is not int or available_weight_version < weight_version:
            raise RuntimeError(
                "version availability wait returned an invalid version: "
                f"requested>={weight_version}, got {available_weight_version!r}"
            )
        tu.assign_non_tensor_data(prompts, "global_steps", available_weight_version)

    def get_last_version_availability_wait_duration_s(self) -> float:
        """Return only the most recent Router availability wait, excluding capacity backpressure."""
        return self._last_version_availability_wait_duration_s

    async def get_metrics(self) -> dict[str, int | float]:
        """Read worker-owned counters without joining fire-and-forget generation tasks."""
        if self.framework_worker is None:
            raise RuntimeError("framework must be initialized before get_metrics")

        return await self.framework_worker.get_metrics.remote()

    def generate_sequences_and_wait(self, prompts) -> None:
        """Blocking variant of :meth:`generate_sequences` for standalone (non-trainer) runs.

        :meth:`generate_sequences` is fire-and-forget (the trainer consumes TQ asynchronously
        via its ReplayBuffer); this awaits the framework worker so the caller knows every
        session's trajectory has landed in TQ, and re-raises any worker-side error.

        Inflight mailbox accounting is intentionally skipped: one joined batch
        keeps the worker mailbox at 1.
        """
        if self.framework_worker is None:
            raise RuntimeError("framework must be initialized before generate_sequences")

        self._wait_until_version_available(prompts)
        ray.get(self.framework_worker.generate_sequences.remote(prompts))
        return None
