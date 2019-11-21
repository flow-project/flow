"""Pending deprecation file.

To view the actual content, go to: flow/networks/base.py
"""
from flow.utils.flow_warnings import deprecated
from flow.networks.base import Network


@deprecated('flow.scenarios.base',
            'flow.networks.base.Network')
class Scenario(Network):
    """See parent class."""

    pass
