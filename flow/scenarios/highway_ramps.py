"""Pending deprecation file.

To view the actual content, go to: flow/networks/highway_ramps.py
"""
from flow.utils.flow_warnings import deprecated
from flow.networks.highway_ramps import HighwayRampsNetwork
from flow.networks.highway_ramps import ADDITIONAL_NET_PARAMS  # noqa: F401


@deprecated('flow.scenarios.highway_ramps',
            'flow.networks.highway_ramps.HighwayRampsNetwork')
class HighwayRampsScenario(HighwayRampsNetwork):
    """See parent class."""

    pass
