"""Pending deprecation file.

To view the actual content, go to: flow/networks/traffic_light_grid.py
"""
from flow.utils.flow_warnings import deprecated
from flow.networks.traffic_light_grid import TrafficLightGridNetwork
from flow.networks.traffic_light_grid import ADDITIONAL_NET_PARAMS  # noqa: F401


@deprecated('flow.scenarios.traffic_light_grid',
            'flow.networks.traffic_light_grid.TrafficLightGridNetwork')
class TrafficLightGridScenario(TrafficLightGridNetwork):
    """See parent class."""

    pass
