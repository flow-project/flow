"""Pending deprecation file.

To view the actual content, go to: flow/envs/traffic_light_grid.py
"""
from flow.utils.flow_warnings import deprecated
from flow.envs.traffic_light_grid import TrafficLightGridEnv as TLGEnv
from flow.envs.traffic_light_grid import TrafficLightGridPOEnv as TLGPOEnv
from flow.envs.traffic_light_grid import TrafficLightGridTestEnv as TLGTEnv


@deprecated('flow.envs.green_wave_env',
            'flow.envs.traffic_light_grid.TrafficLightGridEnv')
class TrafficLightGridEnv(TLGEnv):
    """See parent class."""

    pass


@deprecated('flow.envs.green_wave_env',
            'flow.envs.traffic_light_grid.TrafficLightGridPOEnv')
class PO_TrafficLightGridEnv(TLGPOEnv):
    """See parent class."""

    pass


@deprecated('flow.envs.green_wave_env',
            'flow.envs.traffic_light_grid.TrafficLightGridTestEnv')
class GreenWaveTestEnv(TLGTEnv):
    """See parent class."""

    pass
