"""Empty init file to handle deprecations."""

# base scenario class
from flow.scenarios.base import Scenario

# custom scenarios
from flow.scenarios.bay_bridge import BayBridgeScenario
from flow.scenarios.bay_bridge_toll import BayBridgeTollScenario
from flow.scenarios.bottleneck import BottleneckScenario
from flow.scenarios.figure_eight import FigureEightScenario
from flow.scenarios.traffic_light_grid import TrafficLightGridScenario
from flow.scenarios.highway import HighwayScenario
from flow.scenarios.ring import RingScenario
from flow.scenarios.merge import MergeScenario
from flow.scenarios.multi_ring import MultiRingScenario
from flow.scenarios.minicity import MiniCityScenario
from flow.scenarios.highway_ramps import HighwayRampsScenario

# deprecated classes whose names have changed
from flow.scenarios.figure_eight import Figure8Scenario
from flow.scenarios.loop import LoopScenario
from flow.scenarios.grid import SimpleGridScenario
from flow.scenarios.multi_loop import MultiLoopScenario


__all__ = [
    "Scenario",
    "BayBridgeScenario",
    "BayBridgeTollScenario",
    "BottleneckScenario",
    "FigureEightScenario",
    "TrafficLightGridScenario",
    "HighwayScenario",
    "RingScenario",
    "MergeScenario",
    "MultiRingScenario",
    "MiniCityScenario",
    "HighwayRampsScenario",
    # deprecated classes
    "Figure8Scenario",
    "LoopScenario",
    "SimpleGridScenario",
    "MultiLoopScenario",
]
