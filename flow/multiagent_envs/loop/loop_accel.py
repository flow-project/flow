"""Pending deprecation file.

To view the actual content, go to: flow/envs/multiagent/ring/accel.py
"""
from flow.utils.flow_warnings import deprecated
from flow.envs.multiagent.ring.accel import AdversarialAccelEnv as MAAEnv


@deprecated('flow.multiagent_envs.loop.loop_accel',
            'flow.envs.multiagent.ring.accel.AdversarialAccelEnv')
class AdversarialAccelEnv(MAAEnv):
    """See parent class."""

    pass
