"""Empty init file to ensure documentation for multi-agent envs is created."""

from flow.multiagent_envs.multiagent_env import MultiEnv
from flow.multiagent_envs.ring.wave_attenuation import \
    MultiWaveAttenuationPOEnv
from flow.multiagent_envs.ring.accel import MultiAgentAccelEnv

__all__ = ['MultiEnv', 'MultiAgentAccelEnv', 'MultiWaveAttenuationPOEnv']
