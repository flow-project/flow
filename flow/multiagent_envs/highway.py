"""Pending deprecation file.

To view the actual content, go to: flow/envs/multiagent/highway.py
"""
from flow.utils.flow_warnings import deprecated
from flow.envs.multiagent.highway import MultiAgentHighwayPOEnv as MAHPOEnv
from flow.envs.multiagent.highway import ADDITIONAL_ENV_PARAMS  # noqa: F401


@deprecated('flow.multiagent_envs.highway',
            'flow.envs.multiagent.highway.MultiAgentHighwayPOEnv')
class MultiAgentHighwayPOEnv(MAHPOEnv):
    """See parent class."""

    pass
