"""Contains all callable environments in Flow."""
# flake8: noqa

# single agent envs
from flow.envs.base_env import Env
from flow.envs.bay_bridge.base import BayBridgeEnv
from flow.envs.bottleneck_env import BottleNeckAccelEnv, BottleneckEnv, \
    DesiredVelocityEnv
from flow.envs.green_wave_env import TrafficLightGridEnv, \
    PO_TrafficLightGridEnv, GreenWaveTestEnv
from flow.envs.loop.lane_changing import LaneChangeAccelEnv, \
    LaneChangeAccelPOEnv
from flow.envs.loop.loop_accel import AccelEnv
from flow.envs.loop.loop_merges import TwoLoopsMergePOEnv
from flow.envs.loop.wave_attenuation import WaveAttenuationEnv, \
    WaveAttenuationPOEnv
from flow.envs.merge import WaveAttenuationMergePOEnv
from flow.envs.test import TestEnv

env_list = [
    'Env', 'MultiEnv', 'AccelEnv', 'LaneChangeAccelEnv',
    'LaneChangeAccelPOEnv', 'GreenWaveTestEnv', 'GreenWaveTestEnv',
    'WaveAttenuationMergePOEnv', 'TwoLoopsMergePOEnv', 'BottleneckEnv',
    'BottleNeckAccelEnv', 'WaveAttenuationEnv', 'WaveAttenuationPOEnv',
    'TrafficLightGridEnv', 'PO_TrafficLightGridEnv', 'DesiredVelocityEnv',
    'TestEnv', 'BayBridgeEnv',
]

# multi agent envs
multi_flag = True
try:
    from flow.envs.multiagent_env import MultiEnv
except Exception:
    multi_flag = False

if multi_flag:
    from flow.envs.loop.wave_attenuation import MultiWaveAttenuationPOEnv
    from flow.envs.loop.loop_accel import MultiAgentAccelEnv
    env_list += ['MultiAgentAccelEnv', 'MultiWaveAttenuationPOEnv']
    __all__ = env_list
else:
    __all__ = env_list
