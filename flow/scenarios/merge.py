"""Contains the merge scenario class."""

from flow.scenarios.base_scenario import Scenario
from flow.core.params import InitialConfig
from flow.core.traffic_lights import TrafficLights
from numpy import pi, sin, cos

INFLOW_EDGE_LEN = 100  # length of the inflow edges (needed for resets)
VEHICLE_LENGTH = 5

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

        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)

    def specify_nodes(self, net_params):
        """See parent class."""
        angle = pi / 4
        merge = net_params.additional_params["merge_length"]
        premerge = net_params.additional_params["pre_merge_length"]
        postmerge = net_params.additional_params["post_merge_length"]

        nodes = [
            {
                "id": "inflow_highway",
                "x": repr(-INFLOW_EDGE_LEN),
                "y": repr(0)
            },
            {
                "id": "left",
                "y": repr(0),
                "x": repr(0)
            },
            {
                "id": "center",
                "y": repr(0),
                "x": repr(premerge)
            },
            {
                "id": "right",
                "y": repr(0),
                "x": repr(premerge + postmerge)
            },
            {
                "id": "inflow_merge",
                "x": repr(premerge - (merge + INFLOW_EDGE_LEN) * cos(angle)),
                "y": repr(-(merge + INFLOW_EDGE_LEN) * sin(angle))
            },
            {
                "id": "bottom",
                "x": repr(premerge - merge * cos(angle)),
                "y": repr(-merge * sin(angle))
            },
        ]

        return nodes

    def specify_edges(self, net_params):
        """See parent class."""
        merge = net_params.additional_params["merge_length"]
        premerge = net_params.additional_params["pre_merge_length"]
        postmerge = net_params.additional_params["post_merge_length"]

        edges = [{
            "id": "inflow_highway",
            "type": "highwayType",
            "from": "inflow_highway",
            "to": "left",
            "length": repr(INFLOW_EDGE_LEN)
        }, {
            "id": "left",
            "type": "highwayType",
            "from": "left",
            "to": "center",
            "length": repr(premerge)
        }, {
            "id": "inflow_merge",
            "type": "mergeType",
            "from": "inflow_merge",
            "to": "bottom",
            "length": repr(INFLOW_EDGE_LEN)
        }, {
            "id": "bottom",
            "type": "mergeType",
            "from": "bottom",
            "to": "center",
            "length": repr(merge)
        }, {
            "id": "center",
            "type": "highwayType",
            "from": "center",
            "to": "right",
            "length": repr(postmerge)
        }]

        return edges

    def specify_types(self, net_params):
        """See parent class."""
        h_lanes = net_params.additional_params["highway_lanes"]
        m_lanes = net_params.additional_params["merge_lanes"]
        speed = net_params.additional_params["speed_limit"]

        types = [{
            "id": "highwayType",
            "numLanes": repr(h_lanes),
            "speed": repr(speed)
        }, {
            "id": "mergeType",
            "numLanes": repr(m_lanes),
            "speed": repr(speed)
        }]

        return types

    def specify_routes(self, net_params):
        """See parent class."""
        rts = {
            "inflow_highway": ["inflow_highway", "left", "center"],
            "left": ["left", "center"],
            "center": ["center"],
            "inflow_merge": ["inflow_merge", "bottom", "center"],
            "bottom": ["bottom", "center"]
        }

        return rts

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
