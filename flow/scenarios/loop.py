"""Pending deprecation file.

To view the actual content, go to: flow/scenarios/ring.py
"""
from flow.utils.flow_warnings import deprecated
from flow.scenarios.ring import RingScenario
from flow.scenarios.ring import ADDITIONAL_NET_PARAMS  # noqa: F401


@deprecated('flow.scenarios.loop',
            'flow.scenarios.ring.RingScenario')
class LoopScenario(RingScenario):
    """See parent class."""

    pass
