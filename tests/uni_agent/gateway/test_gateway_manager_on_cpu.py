import asyncio
import os

os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

import httpx
import pytest
import ray

from tests.uni_agent.support import FakeTokenizer, RecordingLLMClient, SequencedBackend


@pytest.fixture(scope="session")
def ray_runtime():
    ray.init(ignore_reinit_error=True, include_dashboard=False)
    yield
    ray.shutdown()


class _FakeRemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _FakeGateway:
    """Minimal gateway stub whose create_session.remote yields at an await, so
    concurrent creates interleave the way real Ray RPCs do."""

    def __init__(self):
        self.created = []

    async def _create(self, session_id, **kwargs):
        await asyncio.sleep(0)  # yield the loop mid-create to expose select/increment races
        self.created.append(session_id)
        return session_id

    @property
    def create_session(self):
        return _FakeRemoteMethod(self._create)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_gateway_manager_balances_concurrent_session_creation():
    """Concurrently created sessions must spread evenly across gateways. The
    selection counter has to be reserved before the create await; otherwise a
    burst of concurrent creates all read the same stale counts and pile onto the
    lowest-index gateway. Builds over fake gateways (no real spawn) to drive the
    routing logic deterministically.
    """
    from uni_agent.gateway.manager import GatewayManager

    gateways = [_FakeGateway() for _ in range(4)]
    # No real Ray spawn: drive the routing logic over injected fakes.
    manager = GatewayManager.__new__(GatewayManager)
    manager.gateways = gateways
    manager.gateway_count = len(gateways)
    manager.active_sessions_per_gateway = [0] * len(gateways)
    manager._session_owners = {}
    manager._llm_client = object()

    await asyncio.gather(*(manager.create_session(f"session-{i}") for i in range(40)))

    counts = manager.active_sessions_per_gateway
    assert sum(counts) == 40
    assert max(counts) - min(counts) <= 1, counts
    assert [len(g.created) for g in gateways] == counts


@pytest.mark.cpu
@pytest.mark.level0
def test_gateway_manager_rejects_zero_gateway_count():
    """``GatewayManager`` raises ``ValueError`` when ``gateway_count=0``, rather
    than silently spawning a half-initialized manager."""
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.manager import GatewayManager

    with pytest.raises(ValueError, match="gateway_count must be positive"):
        GatewayManager(
            llm_client=object(),
            gateway_count=0,
            gateway_actor_config=GatewayActorConfig(tokenizer=object()),
        )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_gateway_manager_round_robins_actors_across_alive_nodes(ray_runtime, monkeypatch):
    """gateway_count > 1 should distribute actors across alive CPU nodes round-robin."""
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.manager import GatewayManager

    fake_nodes = [
        {"NodeID": "a" * 56, "Alive": True, "Resources": {"CPU": 8.0}},
        {"NodeID": "b" * 56, "Alive": True, "Resources": {"CPU": 8.0}},
        {"NodeID": "c" * 56, "Alive": False, "Resources": {"CPU": 8.0}},
        {"NodeID": "d" * 56, "Alive": True, "Resources": {"GPU": 1.0}},
    ]
    monkeypatch.setattr("uni_agent.gateway.manager.ray.nodes", lambda: fake_nodes)

    captured_node_ids = []

    class _StubStartHandle:
        @staticmethod
        def remote():
            return ray.put(None)

    class _StubActorHandle:
        start = _StubStartHandle()

    class _RecordingActor:
        @classmethod
        def options(cls, *, scheduling_strategy):
            captured_node_ids.append(scheduling_strategy.node_id)
            return cls

        @classmethod
        def remote(cls, config, backend):
            return _StubActorHandle()

    monkeypatch.setattr("uni_agent.gateway.gateway.GatewayActor", _RecordingActor)

    manager = GatewayManager(
        llm_client=object(),
        gateway_count=5,
        gateway_actor_config=GatewayActorConfig(tokenizer=object()),
    )

    assert captured_node_ids == ["a" * 56, "b" * 56, "a" * 56, "b" * 56, "a" * 56], captured_node_ids
    assert len(manager.gateways) == 5
    # No shutdown call needed: stub actors have no real Ray state.


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_gateway_manager_finalizes_each_session_on_its_owning_gateway(ray_runtime):
    """create/finalize must route every session back to the same owning gateway
    across a multi-gateway pool. Two sessions land on different gateways and a
    routing-table mix-up would try to finalize an unknown session on the wrong actor.

    Asserts the manager's ownership routing (not the selection policy): any
    placement policy must still finalize a session on the gateway that created it.
    """
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.manager import GatewayManager

    manager = GatewayManager(
        llm_client=RecordingLLMClient("OK"),
        gateway_count=2,
        gateway_actor_config=GatewayActorConfig(tokenizer=FakeTokenizer()),
    )

    session_a = await manager.create_session("session-a")
    session_b = await manager.create_session("session-b")
    # Least-loaded routing places the two sessions on distinct gateways, so a
    # mis-routed finalize would hit the other session's gateway.
    assert session_a.base_url != session_b.base_url

    async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
        for session, label in ((session_a, "a"), (session_b, "b")):
            chat = await client.post(
                f"{session.base_url}/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": f"hi {label}"}]},
            )
            assert chat.status_code == 200

    trajectories_a = await manager.finalize_session("session-a")
    trajectories_b = await manager.finalize_session("session-b")

    assert len(trajectories_a) == 1
    assert len(trajectories_b) == 1

    await manager.shutdown()


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_gateway_manager_default_chains_config_to_http_finalizes_subagent_before_updated_main(
    ray_runtime,
    monkeypatch,
):
    """Exercise default multi-chain behavior from config wiring through HTTP routes."""
    from omegaconf import OmegaConf

    from uni_agent.framework import entry as entry_module

    class _ModelConfig:
        tokenizer = FakeTokenizer()
        processor = None
        hf_config = None

    class _MultiTurnConfig:
        format = None

    class _RolloutConfig:
        name = None
        prompt_length = 2048
        response_length = 2048
        multi_turn = _MultiTurnConfig()

    monkeypatch.setattr(
        entry_module,
        "omega_conf_to_dataclass",
        lambda cfg: _RolloutConfig() if "multi_turn" in cfg else _ModelConfig(),
    )
    config = OmegaConf.create(
        {
            "data": {},
            "actor_rollout_ref": {
                "model": {},
                "rollout": {
                    "prompt_length": 2048,
                    "response_length": 2048,
                    "multi_turn": {"format": None},
                    "custom": {"agent_framework": {"gateway_count": 1}},
                },
            },
        }
    )
    manager = entry_module.build_gateway_manager(
        config=config,
        llm_client=SequencedBackend(["Mango", "Blue", "Apple"]),
    )

    try:
        session = await manager.create_session("session-manager-multiple-chains")
        main_first = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "name a fruit"},
        ]
        subagent = [
            {"role": "system", "content": "You are a focused subagent."},
            {"role": "user", "content": "name a color"},
        ]
        main_continuation = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "name a fruit"},
            {"role": "assistant", "content": "Mango"},
            {"role": "user", "content": "name another fruit"},
        ]

        async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
            for messages in (main_first, subagent, main_continuation):
                chat = await client.post(
                    f"{session.base_url}/chat/completions",
                    json={"model": "m", "messages": messages},
                )
                assert chat.status_code == 200

        trajectories = await manager.finalize_session("session-manager-multiple-chains")

        assert len(trajectories) == 2
        decoded = [FakeTokenizer().decode(trajectory.response_ids) for trajectory in trajectories]
        assert decoded[0] == "Blue"
        assert decoded[1].startswith("Mango")
        assert decoded[1].endswith("Apple")
    finally:
        await manager.shutdown()


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_gateway_manager_allows_independent_http_requests_when_coalescing_disabled(ray_runtime):
    """Opting out lets identical HTTP requests generate concurrently and both materialize."""
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.manager import GatewayManager
    from verl.workers.rollout.replica import TokenOutput

    class _BarrierBackend:
        def __init__(self):
            self.started = 0
            self.both_started = asyncio.Event()

        async def generate(
            self,
            request_id,
            *,
            prompt_ids,
            sampling_params,
            image_data=None,
            video_data=None,
            mm_processor_kwargs=None,
        ):
            response_index = self.started
            self.started += 1
            if self.started == 2:
                self.both_started.set()
            await asyncio.wait_for(self.both_started.wait(), timeout=2)
            text = ["FIRST", "SECOND"][response_index]
            return TokenOutput(
                token_ids=[ord(char) for char in text],
                log_probs=[-0.1] * len(text),
                stop_reason="completed",
            )

    manager = GatewayManager(
        llm_client=_BarrierBackend(),
        gateway_count=1,
        gateway_actor_config=GatewayActorConfig(
            tokenizer=FakeTokenizer(), response_length=128, coalesce_reserved_exact_requests=False
        ),
    )

    try:
        session = await manager.create_session("session-manager-concurrent-http")
        payload = {"model": "m", "messages": [{"role": "user", "content": "same prompt"}]}
        async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
            first, second = await asyncio.gather(
                client.post(f"{session.base_url}/chat/completions", json=payload),
                client.post(f"{session.base_url}/chat/completions", json=payload),
            )

        assert first.status_code == second.status_code == 200
        trajectories = await manager.finalize_session("session-manager-concurrent-http")
        assert sorted(FakeTokenizer().decode(trajectory.response_ids) for trajectory in trajectories) == [
            "FIRST",
            "SECOND",
        ]
    finally:
        await manager.shutdown()


class _LifecycleGateway:
    """Controllable remote-method stub for close failures."""

    def __init__(self, *, finalize_error=None, abort_error=None):
        self.finalize_error = finalize_error
        self.abort_error = abort_error
        self.abort_calls = []
        self.finalize_calls = []

    async def _abort(self, session_id):
        self.abort_calls.append(session_id)
        if self.abort_error is not None:
            raise self.abort_error

    async def _finalize(self, session_id):
        self.finalize_calls.append(session_id)
        if self.finalize_error is not None:
            raise self.finalize_error
        return ["trajectory"]

    abort_session = property(lambda self: _FakeRemoteMethod(self._abort))
    finalize_session = property(lambda self: _FakeRemoteMethod(self._finalize))


def _new_manager(gateway, *, session_ids=(), llm_client=None):
    from uni_agent.gateway.manager import GatewayManager, _SessionOwner

    manager = GatewayManager.__new__(GatewayManager)
    manager.gateways = [gateway]
    manager.active_sessions_per_gateway = [len(session_ids)]
    manager._session_owners = {session_id: _SessionOwner(0, True) for session_id in session_ids}
    manager._llm_client = llm_client or object()
    return manager


class _DetachedRemoteMethod:
    """Ray-like call: cancelling the caller's await does not stop the actor method."""

    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return asyncio.shield(asyncio.ensure_future(self._fn(*args, **kwargs)))


class _SlowCreateGateway(_LifecycleGateway):
    def __init__(self, *, create_error=None):
        super().__init__()
        self.create_error = create_error
        self.created = []
        self.create_started = asyncio.Event()
        self.create_finished = asyncio.Event()
        self.release_create = asyncio.Event()

    async def _create(self, session_id, **kwargs):
        self.create_started.set()
        await self.release_create.wait()
        self.create_finished.set()
        if self.create_error is not None:
            raise self.create_error
        self.created.append(session_id)
        return session_id

    create_session = property(lambda self: _DetachedRemoteMethod(self._create))
    abort_session = property(lambda self: _DetachedRemoteMethod(self._abort))


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("create_error", [None, RuntimeError("create failed")])
async def test_gateway_manager_cancelled_create_aborts_what_the_actor_created(create_error):
    """After the caller cancels, only the manager can still abort the actor-side session."""
    gateway = _SlowCreateGateway(create_error=create_error)
    manager = _new_manager(gateway)

    create = asyncio.create_task(manager.create_session("session-cancelled"))
    await gateway.create_started.wait()
    create.cancel()
    await asyncio.sleep(0)
    gateway.release_create.set()
    with pytest.raises(asyncio.CancelledError):
        await create
    await asyncio.wait_for(gateway.create_finished.wait(), timeout=1)

    assert gateway.abort_calls == ([] if create_error else ["session-cancelled"])
    assert manager._session_owners == {}
    assert manager.active_sessions_per_gateway == [0]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_gateway_manager_cancelled_create_retains_owner_after_abort_failure():
    gateway = _SlowCreateGateway()
    gateway.abort_error = RuntimeError("session abort failed before removal")
    manager = _new_manager(gateway)
    create = asyncio.create_task(manager.create_session("session-cancelled"))
    await gateway.create_started.wait()
    create.cancel()
    gateway.release_create.set()
    with pytest.raises(asyncio.CancelledError):
        await create

    assert set(manager._session_owners) == {"session-cancelled"}
    assert manager._session_owners["session-cancelled"].gateway_index == 0
    assert manager.active_sessions_per_gateway == [1]
    gateway.abort_error = None
    await manager.abort_session("session-cancelled")
    assert gateway.abort_calls == ["session-cancelled"] * 2
    assert manager.active_sessions_per_gateway == [0]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_gateway_manager_repeated_cancellation_still_settles_remote_create():
    gateway = _SlowCreateGateway()
    manager = _new_manager(gateway)
    create = asyncio.create_task(manager.create_session("session-cancelled"))
    await gateway.create_started.wait()
    for _ in range(3):
        create.cancel()
        await asyncio.sleep(0)
    still_waiting = not create.done()
    gateway.release_create.set()
    with pytest.raises(asyncio.CancelledError):
        await create
    await asyncio.wait_for(gateway.create_finished.wait(), timeout=1)

    assert still_waiting
    assert gateway.abort_calls == ["session-cancelled"]
    assert manager._session_owners == {}
    assert manager.active_sessions_per_gateway == [0]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_gateway_manager_close_failure_retains_mapping_and_owner():
    close_error = RuntimeError("actor-close-unknown")
    gateway = _LifecycleGateway(finalize_error=close_error)
    manager = _new_manager(gateway, session_ids=("session-close",))

    with pytest.raises(RuntimeError, match="actor-close-unknown") as raised:
        await manager.finalize_session("session-close")

    assert raised.value is close_error
    assert gateway.finalize_calls == ["session-close"]
    assert set(manager._session_owners) == {"session-close"}
    assert manager._session_owners["session-close"].gateway_index == 0
    assert manager.active_sessions_per_gateway == [1]

    await manager.abort_session("session-close")

    assert gateway.abort_calls == ["session-close"]
    assert manager._session_owners == {}
    assert manager.active_sessions_per_gateway == [0]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_gateway_manager_rejects_duplicate_create_without_touching_owner():
    gateway = _SlowCreateGateway()
    manager = _new_manager(gateway)
    create = asyncio.create_task(manager.create_session("same-id"))
    await gateway.create_started.wait()

    with pytest.raises(RuntimeError, match="already exists"):
        await manager.create_session("same-id")
    assert set(manager._session_owners) == {"same-id"}
    assert manager._session_owners["same-id"].gateway_index == 0
    assert manager.active_sessions_per_gateway == [1]
    assert gateway.abort_calls == []

    gateway.release_create.set()
    assert await create == "same-id"
    await manager.abort_session("same-id")
    assert manager.active_sessions_per_gateway == [0]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_gateway_manager_concurrent_closes_release_load_slot_once():
    class _ConcurrentCloseGateway(_LifecycleGateway):
        async def _finalize(self, session_id):
            await asyncio.sleep(0)
            return await super()._finalize(session_id)

        async def _abort(self, session_id):
            await asyncio.sleep(0)
            await super()._abort(session_id)

    gateway = _ConcurrentCloseGateway()
    manager = _new_manager(gateway, session_ids=("session-close",))
    await asyncio.gather(manager.finalize_session("session-close"), manager.abort_session("session-close"))
    assert manager._session_owners == {}
    assert manager.active_sessions_per_gateway == [0]


class _RouteClient:
    """Return non-coroutine awaitables, as Ray-backed routing clients do."""

    def __init__(self, bound_version=7, *, bind_error=None, release_errors=()):
        self.bound_version = bound_version
        self.bind_error = bind_error
        self.release_errors = list(release_errors)
        self.bind_calls = []
        self.release_calls = []
        self.bind_started = asyncio.Event()
        self.release_started = asyncio.Event()
        self.allow_bind = asyncio.Event()
        self.allow_release = asyncio.Event()
        self.allow_bind.set()
        self.allow_release.set()

    def bind_route(self, **kwargs):
        async def bind():
            self.bind_calls.append(kwargs)
            self.bind_started.set()
            await self.allow_bind.wait()
            if self.bind_error is not None:
                raise self.bind_error
            return self.bound_version

        return asyncio.ensure_future(bind())

    def release_route(self, **kwargs):
        async def release():
            self.release_calls.append(kwargs)
            self.release_started.set()
            await self.allow_release.wait()
            if self.release_errors:
                raise self.release_errors.pop(0)

        return asyncio.ensure_future(release())


class _RouteGateway(_SlowCreateGateway):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.create_kwargs = []
        self.release_create.set()

    async def _create(self, session_id, **kwargs):
        self.create_kwargs.append(kwargs)
        return await super()._create(session_id, **kwargs)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("requested,bound", [(3, 7), (3, None), (None, 7)])
async def test_gateway_manager_passes_only_final_bound_version_to_actor(requested, bound):
    backend = _RouteClient(bound)
    gateway = _RouteGateway()
    manager = _new_manager(gateway, llm_client=backend)
    await manager.create_session("bound", weight_version=requested, metadata={"owner": "runner"}, sampling_params={})
    assert backend.bind_calls == [{"session_id": "bound", "weight_version": requested}]
    assert gateway.create_kwargs == [
        {
            "weight_version": bound if requested is not None else None,
            "metadata": {"owner": "runner"},
            "sampling_params": {},
        }
    ]
    await manager.abort_session("bound")
    assert backend.release_calls == [{"session_id": "bound"}]
    assert manager.active_sessions_per_gateway == [0]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_gateway_manager_plain_backend_does_not_pin_batch_version():
    gateway = _RouteGateway()
    manager = _new_manager(gateway)
    await manager.create_session("plain", weight_version=3)
    assert gateway.create_kwargs == [{"weight_version": None}]
    await manager.abort_session("plain")


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["bind", "actor", "stale-version", "invalid-version"])
@pytest.mark.parametrize("release_fails", [False, True])
async def test_gateway_manager_failed_create_keeps_original_error_and_releases_only_owned_route(stage, release_fails):
    original = RuntimeError("original create failure")
    backend = _RouteClient(
        {"stale-version": 2, "invalid-version": True}.get(stage, 7),
        bind_error=original if stage == "bind" else None,
        release_errors=[RuntimeError("cleanup failed")] if release_fails else [],
    )
    gateway = _RouteGateway(create_error=original if stage == "actor" else None)
    manager = _new_manager(gateway, llm_client=backend)
    error_type = ValueError if stage.endswith("version") else RuntimeError
    with pytest.raises(error_type) as raised:
        await manager.create_session("failed", weight_version=3)
    if error_type is RuntimeError:
        assert raised.value is original
    assert backend.release_calls == [{"session_id": "failed"}]
    assert gateway.abort_calls == []
    if release_fails:
        assert not manager._session_owners["failed"].actor_created
        await manager.abort_session("failed")
        assert gateway.abort_calls == []  # A failed actor create may refer to a pre-existing session.
    assert manager._session_owners == {}
    assert manager.active_sessions_per_gateway == [0]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("failures", [1, 2])
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
async def test_gateway_manager_release_failure_keeps_finalized_trajectories_and_retry_owner(failures, error_type):
    backend = _RouteClient(release_errors=[error_type("release failed") for _ in range(failures)])
    gateway = _RouteGateway()
    manager = _new_manager(gateway, session_ids=("finished",), llm_client=backend)
    assert await manager.finalize_session("finished") == ["trajectory"]
    assert len(backend.release_calls) == 2
    assert gateway.abort_calls == []
    if failures == 2:
        assert not manager._session_owners["finished"].actor_created
        assert manager.active_sessions_per_gateway == [1]
        await manager.abort_session("finished")
        assert gateway.abort_calls == []
    assert manager.active_sessions_per_gateway == [0]
    assert manager._session_owners == {}


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["bind", "create"])
async def test_gateway_manager_cancelled_route_creation_settles_before_owned_abort(phase):
    backend = _RouteClient()
    gateway = _RouteGateway()
    if phase == "bind":
        backend.allow_bind.clear()
        started, allow = backend.bind_started, backend.allow_bind
    else:
        gateway.release_create.clear()
        started, allow = gateway.create_started, gateway.release_create
    manager = _new_manager(gateway, llm_client=backend)
    create = asyncio.create_task(manager.create_session("cancelled", weight_version=3))
    await started.wait()
    for _ in range(3):
        create.cancel()
        await asyncio.sleep(0)
    assert not create.done()
    allow.set()
    with pytest.raises(asyncio.CancelledError):
        await create
    assert gateway.abort_calls == ["cancelled"]
    assert backend.release_calls == [{"session_id": "cancelled"}]
    assert manager.active_sessions_per_gateway == [0]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("close", ["finalize_session", "abort_session"])
async def test_gateway_manager_repeated_cancellation_settles_route_release_then_propagates(close):
    backend = _RouteClient()
    backend.allow_release.clear()
    gateway = _RouteGateway()
    manager = _new_manager(gateway, session_ids=("closing",), llm_client=backend)
    closing = asyncio.create_task(getattr(manager, close)("closing"))
    await backend.release_started.wait()
    for _ in range(3):
        closing.cancel()
        await asyncio.sleep(0)
    assert not closing.done()
    backend.allow_release.set()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert backend.release_calls == [{"session_id": "closing"}]
    assert manager.active_sessions_per_gateway == [0]
    assert manager._session_owners == {}


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("requested", [-1, True, 1.5])
async def test_gateway_manager_invalid_requested_version_has_no_lifecycle_side_effects(requested):
    backend = _RouteClient()
    gateway = _RouteGateway()
    manager = _new_manager(gateway, llm_client=backend)
    with pytest.raises(ValueError, match="non-negative integer"):
        await manager.create_session("invalid", weight_version=requested)
    assert backend.bind_calls == backend.release_calls == gateway.create_kwargs == []
    assert manager._session_owners == {}
    assert manager.active_sessions_per_gateway == [0]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_gateway_manager_bind_error_survives_cancelled_release():
    original = RuntimeError("bind failed")
    backend = _RouteClient(bind_error=original, release_errors=[asyncio.CancelledError()])
    manager = _new_manager(_RouteGateway(), llm_client=backend)
    with pytest.raises(RuntimeError) as raised:
        await manager.create_session("failed", weight_version=3)
    assert raised.value is original
    assert not manager._session_owners["failed"].actor_created
    await manager.abort_session("failed")
    assert manager._session_owners == {}


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("failures", [1, 2])
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
async def test_gateway_manager_cancelled_finalize_propagates_even_when_release_fails(failures, error_type):
    backend = _RouteClient(release_errors=[error_type("release failed") for _ in range(failures)])
    backend.allow_release.clear()
    manager = _new_manager(_RouteGateway(), session_ids=("closing",), llm_client=backend)
    closing = asyncio.create_task(manager.finalize_session("closing"))
    await backend.release_started.wait()
    closing.cancel()
    await asyncio.sleep(0)
    backend.allow_release.set()
    with pytest.raises(asyncio.CancelledError):
        await closing
    if failures == 2:
        assert not manager._session_owners["closing"].actor_created
        await manager.abort_session("closing")
    assert manager._session_owners == {}
    assert manager.active_sessions_per_gateway == [0]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_gateway_manager_cancelled_finalize_settles_actor_before_route_release():
    started, finish = asyncio.Event(), asyncio.Event()

    class _SlowFinalizeGateway(_RouteGateway):
        async def _finalize(self, session_id):
            started.set()
            await finish.wait()
            return await super()._finalize(session_id)

    backend = _RouteClient()
    gateway = _SlowFinalizeGateway()
    manager = _new_manager(gateway, session_ids=("closing",), llm_client=backend)
    closing = asyncio.create_task(manager.finalize_session("closing"))
    await started.wait()
    closing.cancel()
    await asyncio.sleep(0)
    assert not closing.done() and backend.release_calls == []
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert backend.release_calls == [{"session_id": "closing"}]
    assert manager._session_owners == {}


class _SharedRoutingClient:
    def __init__(self, authority):
        self.authority = authority

    def bind_route(self, **kwargs):
        return self.authority.bind.remote(**kwargs)

    def release_route(self, **kwargs):
        return self.authority.release.remote(**kwargs)

    async def generate(self, request_id, *, prompt_ids, sampling_params, image_data=None, video_data=None, **kwargs):
        assert "weight_version" not in kwargs
        return await self.authority.generate.remote(session_id=request_id)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["pinned", "conflicting", "latest"])
async def test_gateway_manager_route_authority_is_shared_across_ray_client_copies(ray_runtime, mode):
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.manager import GatewayManager
    from verl.workers.rollout.replica import TokenOutput

    @ray.remote
    class _Authority:
        def __init__(self):
            self.routes = {}
            self.served_version = 8 if mode == "conflicting" else 7

        def bind(self, session_id, weight_version):
            self.routes[session_id] = 7
            return None if mode == "latest" else 7

        def generate(self, session_id):
            assert session_id in self.routes
            version = self.served_version
            if mode == "latest":
                self.served_version += 1
            return TokenOutput(
                token_ids=[ord("O"), ord("K")],
                log_probs=[-0.1, -0.1],
                stop_reason="completed",
                extra_fields={"min_global_steps": version, "max_global_steps": version},
            )

        def release(self, session_id):
            self.routes.pop(session_id, None)

        def remaining(self):
            return list(self.routes)

    authority = _Authority.remote()
    manager = GatewayManager(
        _SharedRoutingClient(authority),
        gateway_count=1,
        gateway_actor_config=GatewayActorConfig(tokenizer=FakeTokenizer()),
    )
    try:
        session = await manager.create_session("shared", weight_version=3)
        messages = [{"role": "user", "content": "prompt"}]
        async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
            response = await client.post(f"{session.base_url}/chat/completions", json={"messages": messages})
            if mode == "conflicting":
                assert response.status_code == 500
                await manager.abort_session("shared")
            else:
                assert response.status_code == 200
                if mode == "latest":
                    messages += [{"role": "assistant", "content": "OK"}, {"role": "user", "content": "again"}]
                    response = await client.post(f"{session.base_url}/chat/completions", json={"messages": messages})
                    assert response.status_code == 200
                [trajectory] = await manager.finalize_session("shared")
                assert trajectory.extra_fields["min_global_steps"] == 7
                assert trajectory.extra_fields["max_global_steps"] == (8 if mode == "latest" else 7)
        assert await authority.remaining.remote() == []
        assert manager.active_sessions_per_gateway == [0]
    finally:
        await manager.shutdown()
        ray.kill(authority)
