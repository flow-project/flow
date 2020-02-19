"""Empty init file to ensure documentation for multi-agent envs is created."""

from flow.multiagent_envs.multiagent_env import MultiEnv
from flow.multiagent_envs.loop.wave_attenuation import \
    MultiWaveAttenuationPOEnv
from flow.multiagent_envs.loop.loop_accel import AdversarialAccelEnv
from flow.multiagent_envs.traffic_light_grid import MultiTrafficLightGridPOEnv
from flow.multiagent_envs.highway import MultiAgentHighwayPOEnv

__all__ = [
    'MultiEnv',
    'AdversarialAccelEnv',
    'MultiWaveAttenuationPOEnv',
    'MultiTrafficLightGridPOEnv',
    'MultiAgentHighwayPOEnv'
]
