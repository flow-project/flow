"""Pending deprecation file.

To view the actual content, go to: flow/scenarios/multi_ring.py
"""
from flow.utils.flow_warnings import deprecated
from flow.scenarios.multi_ring import MultiRingScenario
from flow.scenarios.multi_ring import ADDITIONAL_NET_PARAMS  # noqa: F401


@deprecated('flow.scenarios.multi_loop',
            'flow.scenarios.multi_ring.MultiRingScenario')
class MultiLoopScenario(MultiRingScenario):
    """See parent class."""

    pass
