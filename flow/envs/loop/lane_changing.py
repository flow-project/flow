"""Pending deprecation file.

To view the actual content, go to: flow/envs/ring/lane_change_accel.py
"""
from flow.utils.flow_warnings import deprecated
from flow.envs.ring.lane_change_accel import LaneChangeAccelEnv as LCEnv
from flow.envs.ring.lane_change_accel import LaneChangeAccelPOEnv as LCPOEnv


@deprecated('flow.envs.loop.lane_changing',
            'flow.envs.ring.lane_change_accel.LaneChangeAccelEnv')
class LaneChangeAccelEnv(LCEnv):
    """See parent class."""

    pass


@deprecated('flow.envs.loop.lane_changing',
            'flow.envs.ring.lane_change_accel.LaneChangeAccelPOEnv')
class LaneChangeAccelPOEnv(LCPOEnv):
    """See parent class."""

    pass
