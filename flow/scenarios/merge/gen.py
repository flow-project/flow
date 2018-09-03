"""Contains the merge generator class."""

from flow.core.generator import Generator

from numpy import pi, sin, cos

INFLOW_EDGE_LEN = 100  # length of the inflow edges (needed for resets)
VEHICLE_LENGTH = 5


class MergeGenerator(Generator):
    """Generator for merge networks."""

    def __init__(self, net_params, base):
        merge = net_params.additional_params["merge_length"]
        premerge = net_params.additional_params["pre_merge_length"]
        postmerge = net_params.additional_params["post_merge_length"]
        length = merge + premerge + postmerge + 2 * INFLOW_EDGE_LEN + 8.1
        h_lanes = net_params.additional_params["highway_lanes"]
        m_lanes = net_params.additional_params["merge_lanes"]
        self.name = "{}-{}m{}l-{}l".format(base, length, h_lanes, m_lanes)
        super().__init__(net_params, base)

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
