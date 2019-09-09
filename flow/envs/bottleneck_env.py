"""Pending deprecation file.

To view the actual content, go to: flow/envs/bottleneck.py
"""
from flow.utils.flow_warnings import deprecated
from flow.envs.bottleneck import BottleneckEnv as BEnv
from flow.envs.bottleneck import BottleneckAccelEnv as BAEnv
from flow.envs.bottleneck import BottleneckDesiredVelocityEnv as BDVEnv


@deprecated('flow.envs.bottleneck_env',
            'flow.envs.bottleneck.BottleneckEnv')
class BottleneckEnv(BEnv):
    """See parent class."""

    pass


@deprecated('flow.envs.bottleneck_env',
            'flow.envs.bottleneck.BottleneckAccelEnv')
class BottleNeckAccelEnv(BAEnv):
    """See parent class."""

    pass


@deprecated('flow.envs.bottleneck_env',
            'flow.envs.bottleneck.BottleneckDesiredVelocityEnv')
class DesiredVelocityEnv(BDVEnv):
    """See parent class."""

    pass
