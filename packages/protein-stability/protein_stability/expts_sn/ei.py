from aviary_internal.expts.sn.sft import LocalAgentConfig

from .eval import StabilitySimpleAgentRolloutExpt


class StabilityLocalRolloutExpt(StabilitySimpleAgentRolloutExpt):
    agent: LocalAgentConfig  # type: ignore[assignment]
