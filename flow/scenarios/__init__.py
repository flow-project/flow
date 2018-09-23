"""Contains all available scenarios and generators in Flow."""

# base scenario class
from flow.scenarios.base_scenario import Scenario

# custom generators
from flow.scenarios.bay_bridge.gen import BayBridgeGenerator
from flow.scenarios.bay_bridge_toll.gen import BayBridgeTollGenerator
from flow.scenarios.bottleneck.gen import BottleneckGenerator
from flow.scenarios.figure8.gen import Figure8Generator
from flow.scenarios.grid.gen import SimpleGridGenerator
from flow.scenarios.highway.gen import HighwayGenerator
from flow.scenarios.loop.gen import CircleGenerator
from flow.scenarios.merge.gen import MergeGenerator
from flow.scenarios.netfile.gen import NetFileGenerator
from flow.scenarios.loop_merge.gen import TwoLoopOneMergingGenerator

# custom scenarios
from flow.scenarios.bay_bridge.scenario import BayBridgeScenario
from flow.scenarios.bay_bridge_toll.scenario import BayBridgeTollScenario
from flow.scenarios.bottleneck.scenario import BottleneckScenario
from flow.scenarios.figure8.figure8_scenario import Figure8Scenario
from flow.scenarios.grid.grid_scenario import SimpleGridScenario
from flow.scenarios.highway.scenario import HighwayScenario
from flow.scenarios.loop.loop_scenario import LoopScenario
from flow.scenarios.merge.scenario import MergeScenario
from flow.scenarios.netfile.scenario import NetFileScenario
from flow.scenarios.loop_merge.scenario import TwoLoopsOneMergingScenario

__all__ = [
    "Scenario", "BayBridgeGenerator", "BayBridgeTollGenerator",
    "BottleneckGenerator", "Figure8Generator", "SimpleGridGenerator",
    "HighwayGenerator", "CircleGenerator", "MergeGenerator",
    "NetFileGenerator", "TwoLoopOneMergingGenerator", "BayBridgeScenario",
    "BayBridgeTollScenario", "BottleneckScenario", "Figure8Scenario",
    "SimpleGridScenario", "HighwayScenario", "LoopScenario", "MergeScenario",
    "NetFileScenario", "TwoLoopsOneMergingScenario"
]
