"""Pending deprecation file.

To view the actual content, go to: flow/envs/multiagent/traffic_light_grid.py
"""
from flow.utils.flow_warnings import deprecated
from flow.envs.multiagent.ring.accel import MultiAgentAccelEnv as MAAEnv


@deprecated('flow.multiagent_envs.loop.loop_accel',
            'flow.envs.multiagent.ring.accel.MultiAgentAccelEnv')
class MultiAgentAccelEnv(MAAEnv):
    """See parent class."""

    pass
