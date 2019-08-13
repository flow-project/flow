"""Contains all available scenarios in Flow."""

# base scenario class
from flow.scenarios.base_scenario import Scenario

# custom scenarios
from flow.scenarios.bay_bridge import BayBridgeScenario
from flow.scenarios.bay_bridge_toll import BayBridgeTollScenario
from flow.scenarios.bottleneck import BottleneckScenario
from flow.scenarios.figure_eight import Figure8Scenario
from flow.scenarios.grid import SimpleGridScenario
from flow.scenarios.highway import HighwayScenario
from flow.scenarios.loop import LoopScenario
from flow.scenarios.merge import MergeScenario
from flow.scenarios.loop_merge import TwoLoopsOneMergingScenario
from flow.scenarios.multi_loop import MultiLoopScenario
from flow.scenarios.minicity import MiniCityScenario
from flow.scenarios.udssc import UDSSCMergingScenario
from flow.scenarios.highway_ramps import HighwayRampsScenario

__all__ = [
    "Scenario", "BayBridgeScenario", "BayBridgeTollScenario",
    "BottleneckScenario", "Figure8Scenario", "SimpleGridScenario",
    "HighwayScenario", "LoopScenario", "MergeScenario",
    "TwoLoopsOneMergingScenario", "MultiLoopScenario", "MiniCityScenario",
    "HighwayRampsScenario", "UDSSCMergingScenario"
]
