from __future__ import annotations

from abc import ABC, abstractmethod

from tensordict import TensorDict


class AgentFramework(ABC):
    """Abstract base for trainer-driven agent frameworks."""

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        *,
        config,
        **kwargs,
    ) -> AgentFramework: ...

    @abstractmethod
    async def generate_sequences(self, prompts: TensorDict) -> None:
        """Run agent sessions and write finalized trajectories to TransferQueue."""
        ...

    def get_metrics(self) -> dict[str, int | float]:
        """Return cumulative framework metrics for the rollout Tracker."""
        raise NotImplementedError(f"{type(self).__name__} does not expose framework metrics")
