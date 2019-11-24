"""Pending deprecation file.

To view the actual content, go to: flow/envs/multiagent/traffic_light_grid.py
"""
from flow.utils.flow_warnings import deprecated
from flow.envs.multiagent.traffic_light_grid import MultiTrafficLightGridPOEnv as MTLGPOEnv
from flow.envs.multiagent.traffic_light_grid import ADDITIONAL_ENV_PARAMS  # noqa: F401


@deprecated('flow.multiagent_envs.traffic_light_grid',
            'flow.envs.multiagent.traffic_light_grid.MultiEnv')
class MultiTrafficLightGridPOEnv(MTLGPOEnv):
    """See parent class."""

    pass
