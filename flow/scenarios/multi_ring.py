"""Pending deprecation file.

To view the actual content, go to: flow/networks/multi_ring.py
"""
from flow.utils.flow_warnings import deprecated
from flow.networks.multi_ring import MultiRingNetwork
from flow.networks.multi_ring import ADDITIONAL_NET_PARAMS  # noqa: F401


@deprecated('flow.scenarios.multi_ring',
            'flow.networks.multi_ring.RingNetwork')
class MultiRingScenario(MultiRingNetwork):
    """See parent class."""

    pass
