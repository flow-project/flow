"""Pending deprecation file.

To view the actual content, go to: flow/networks/figure_eight.py
"""
from flow.utils.flow_warnings import deprecated
from flow.networks.figure_eight import FigureEightNetwork
from flow.networks.figure_eight import ADDITIONAL_NET_PARAMS  # noqa: F401


@deprecated('flow.scenarios.figure_eight',
            'flow.networks.figure_eight.FigureEightNetwork')
class FigureEightScenario(FigureEightNetwork):
    """See parent class."""

    pass


@deprecated('flow.scenarios.figure_eight',
            'flow.networks.figure_eight.FigureEightNetwork')
class Figure8Scenario(FigureEightNetwork):
    """See parent class."""

    pass
