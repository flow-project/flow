"""Pending deprecation file.

To view the actual content, go to: flow/scenarios/base.py
"""
from flow.utils.flow_warnings import deprecated
from flow.scenarios.base import Scenario as Scen


@deprecated('flow.scenarios.base_scenario',
            'flow.scenarios.base.Scenario')
class Scenario(Scen):
    """See parent class."""

    pass
