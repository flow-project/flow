"""Empty init file to ensure documentation for multi-agent envs is created."""

from flow.envs.multiagent.base import MultiEnv
from flow.envs.multiagent.ring.wave_attenuation import \
    MultiWaveAttenuationPOEnv
from flow.envs.multiagent.ring.accel import MultiAgentAccelEnv

__all__ = ['MultiEnv', 'MultiAgentAccelEnv', 'MultiWaveAttenuationPOEnv']
