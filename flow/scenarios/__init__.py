"""Contains all available scenarios in Flow."""

# base scenario class
from flow.scenarios.base import Scenario

# custom scenarios
from flow.scenarios.bay_bridge import BayBridgeScenario
from flow.scenarios.bay_bridge_toll import BayBridgeTollScenario
from flow.scenarios.bottleneck import BottleneckScenario
from flow.scenarios.figure_eight import Figure8Scenario
from flow.scenarios.grid import SimpleGridScenario
from flow.scenarios.highway import HighwayScenario
from flow.scenarios.ring import RingScenario
from flow.scenarios.merge import MergeScenario
from flow.scenarios.ring_merge import TwoRingsOneMergingScenario
from flow.scenarios.multi_ring import MultiRingScenario
from flow.scenarios.minicity import MiniCityScenario

__all__ = [
    "Scenario", "BayBridgeScenario", "BayBridgeTollScenario",
    "BottleneckScenario", "Figure8Scenario", "SimpleGridScenario",
    "HighwayScenario", "RingScenario", "MergeScenario",
    "TwoRingsOneMergingScenario", "MultiRingScenario", "MiniCityScenario"
]
