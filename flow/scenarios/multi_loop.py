"""Pending deprecation file.

To view the actual content, go to: flow/networks/multi_ring.py
"""
from flow.utils.flow_warnings import deprecated
from flow.networks.multi_ring import MultiRingNetwork
from flow.networks.multi_ring import ADDITIONAL_NET_PARAMS  # noqa: F401


@deprecated('flow.scenarios.multi_loop',
            'flow.networks.multi_ring.MultiRingNetwork')
class MultiLoopScenario(MultiRingNetwork):
    """See parent class."""

    pass
