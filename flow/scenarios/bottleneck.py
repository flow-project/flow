"""Pending deprecation file.

To view the actual content, go to: flow/networks/bottleneck.py
"""
from flow.utils.flow_warnings import deprecated
from flow.networks.bottleneck import BottleneckNetwork
from flow.networks.bottleneck import ADDITIONAL_NET_PARAMS  # noqa: F401


@deprecated('flow.scenarios.bottleneck',
            'flow.networks.bottleneck.BottleneckNetwork')
class BottleneckScenario(BottleneckNetwork):
    """See parent class."""

    pass
