"""Contains the merge scenario class."""

from flow.scenarios.base_scenario import Scenario
from flow.core.params import InitialConfig
from flow.core.traffic_lights import TrafficLights
from flow.scenarios.merge.gen import INFLOW_EDGE_LEN

ADDITIONAL_NET_PARAMS = {
    # length of the merge edge
    "merge_length": 100,
    # length of the highway leading to the merge
    "pre_merge_length": 200,
    # length of the highway past the merge
    "post_merge_length": 100,
    # number of lanes in the merge
    "merge_lanes": 1,
    # number of lanes in the highway
    "highway_lanes": 1,
    # max speed limit of the network
    "speed_limit": 30,
}


class MergeScenario(Scenario):
    """Scenario class for highways with a single in-merge."""

    def __init__(self,
                 name,
                 generator_class,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLights()):
        """Initialize a merge scenario.

        Requires from net_params:
        - merge_length: length of the merge edge
        - pre_merge_length: length of the highway leading to the merge
        - post_merge_length: length of the highway past the merge
        - merge_lanes: number of lanes in the merge
        - highway_lanes: number of lanes in the highway
        - speed_limit: max speed limit of the network

        See flow/scenarios/base_scenario.py for description of params.
        """
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        super().__init__(name, generator_class, vehicles, net_params,
                         initial_config, traffic_lights)

    def specify_edge_starts(self):
        """See parent class."""
        premerge = self.net_params.additional_params["pre_merge_length"]
        postmerge = self.net_params.additional_params["post_merge_length"]

        edgestarts = [("inflow_highway", 0), ("left", INFLOW_EDGE_LEN + 0.1),
                      ("center", INFLOW_EDGE_LEN + premerge + 8.1),
                      ("inflow_merge",
                       INFLOW_EDGE_LEN + premerge + postmerge + 8.1),
                      ("bottom",
                       2 * INFLOW_EDGE_LEN + premerge + postmerge + 8.2)]

        return edgestarts

    def specify_internal_edge_starts(self):
        """See parent class."""
        premerge = self.net_params.additional_params["pre_merge_length"]
        postmerge = self.net_params.additional_params["post_merge_length"]

        internal_edgestarts = [
            (":left", INFLOW_EDGE_LEN), (":center",
                                         INFLOW_EDGE_LEN + premerge + 0.1),
            (":bottom", 2 * INFLOW_EDGE_LEN + premerge + postmerge + 8.1)
        ]

        return internal_edgestarts
