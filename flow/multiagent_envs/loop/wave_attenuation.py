"""Pending deprecation file.

To view the actual content, go to: flow/envs/multiagent/traffic_light_grid.py
"""
from flow.utils.flow_warnings import deprecated
from flow.envs.multiagent.ring.wave_attenuation import MultiWaveAttenuationPOEnv as MWAPOEnv
from flow.envs.multiagent.ring.wave_attenuation import ADDITIONAL_ENV_PARAMS  # noqa: F401


@deprecated('flow.multiagent_envs.loop.wave_attenuation',
            'flow.envs.multiagent.ring.wave_attenuation.MultiWaveAttenuationPOEnv')
class MultiWaveAttenuationPOEnv(MWAPOEnv):
    """See parent class."""

    pass
