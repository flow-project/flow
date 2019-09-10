"""Pending deprecation file.

To view the actual content, go to: flow/envs/ring/accel.py
"""
from flow.utils.flow_warnings import deprecated
from flow.envs.ring.accel import AccelEnv as AEnv


@deprecated('flow.envs.loop.accel',
            'flow.envs.ring.accel.AccelEnv')
class AccelEnv(AEnv):
    """See parent class."""

    pass
