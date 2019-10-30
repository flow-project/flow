"""Pending deprecation file.

To view the actual content, go to: flow/networks/ring.py
"""
from flow.utils.flow_warnings import deprecated
from flow.networks.ring import RingNetwork
from flow.networks.ring import ADDITIONAL_NET_PARAMS  # noqa: F401


@deprecated('flow.scenarios.ring',
            'flow.networks.ring.RingNetwork')
class RingScenario(RingNetwork):
    """See parent class."""

    pass
