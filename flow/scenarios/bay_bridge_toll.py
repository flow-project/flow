"""Pending deprecation file.

To view the actual content, go to: flow/networks/bay_bridge_toll.py
"""
from flow.utils.flow_warnings import deprecated
from flow.networks.bay_bridge_toll import BayBridgeTollNetwork


@deprecated('flow.scenarios.bay_bridge_toll',
            'flow.networks.bay_bridge_toll.BayBridgeTollNetwork')
class BayBridgeTollScenario(BayBridgeTollNetwork):
    """See parent class."""

    pass
