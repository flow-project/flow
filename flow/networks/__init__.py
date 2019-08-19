"""Contains all available scenarios in Flow."""

# base scenario class
from flow.networks.base_scenario import Network

# custom scenarios
from flow.networks.bay_bridge import BayBridgeNetwork
from flow.networks.bay_bridge_toll import BayBridgeTollNetwork
from flow.networks.bottleneck import BottleneckNetwork
from flow.networks.figure_eight import Figure8Network
from flow.networks.grid import SimpleGridNetwork
from flow.networks.highway import HighwayNetwork
from flow.networks.loop import LoopNetwork
from flow.networks.merge import MergeNetwork
from flow.networks.loop_merge import TwoLoopsOneMergingNetwork
from flow.networks.multi_loop import MultiLoopNetwork
from flow.networks.minicity import MiniCityNetwork
from flow.networks.highway_ramps import HighwayRampsNetwork

__all__ = [
    "Network", "BayBridgeNetwork", "BayBridgeTollNetwork",
    "BottleneckNetwork", "Figure8Network", "SimpleGridNetwork",
    "HighwayNetwork", "LoopNetwork", "MergeNetwork",
    "TwoLoopsOneMergingNetwork", "MultiLoopNetwork", "MiniCityNetwork",
    "HighwayRampsNetwork"
]
