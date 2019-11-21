"""Pending deprecation file.

To view the actual content, go to: flow/networks/minicity.py
"""
from flow.utils.flow_warnings import deprecated
from flow.networks.minicity import MiniCityNetwork


@deprecated('flow.scenarios.minicity',
            'flow.networks.minicity.MiniCityNetwork')
class MiniCityScenario(MiniCityNetwork):
    """See parent class."""

    pass
