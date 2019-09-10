"""Pending deprecation file.

To view the actual content, go to: flow/scenarios/traffic_light_grid.py
"""
from flow.utils.flow_warnings import deprecated
from flow.scenarios.traffic_light_grid import TrafficLightGridScenario
from flow.scenarios.traffic_light_grid import ADDITIONAL_NET_PARAMS  # noqa: F401


@deprecated('flow.scenarios.grid',
            'flow.scenarios.traffic_light_grid.TrafficLightGridScenario')
class SimpleGridScenario(TrafficLightGridScenario):
    """See parent class."""

    pass
