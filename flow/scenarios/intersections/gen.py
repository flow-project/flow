from flow.core.generator import Generator

from lxml import etree

E = etree.Element


class TwoWayIntersectionGenerator(Generator):
    """
    Generator for two-way intersections.
    """
    def __init__(self, net_params, base):
        """
        See parent class
        """
        super().__init__(net_params, base)

        horz_length_in = net_params.additional_params["horizontal_length_in"]
        horz_length_out = net_params.additional_params["horizontal_length_out"]
        horz_lanes = net_params.additional_params["horizontal_lanes"]
        vert_length_in = net_params.additional_params["vertical_length_in"]
        vert_length_out = net_params.additional_params["vertical_length_out"]
        vert_lanes = net_params.additional_params["vertical_lanes"]

        self.name = "%s-horizontal-%dm%dl-vertical-%dm%dl" % \
                    (base, horz_length_in + horz_length_out, horz_lanes,
                     vert_length_in + vert_length_out, vert_lanes)

    def specify_nodes(self, net_params):
        """
        See parent class
        """
        horz_length_in = net_params.additional_params["horizontal_length_in"]
        horz_length_out = net_params.additional_params["horizontal_length_out"]
        vert_length_in = net_params.additional_params["vertical_length_in"]
        vert_length_out = net_params.additional_params["vertical_length_out"]

        nodes = [{"id": "center", "x": repr(0), "y": repr(0),
                  "type": "priority"},
                 {"id": "bottom", "x": repr(0), "y": repr(-vert_length_in),
                  "type": "priority"},
                 {"id": "top",    "x": repr(0), "y": repr(vert_length_out),
                  "type": "priority"},
                 {"id": "left",   "x": repr(-horz_length_in), "y": repr(0),
                  "type": "priority"},
                 {"id": "right",  "x": repr(horz_length_out), "y": repr(0),
                  "type": "priority"}]

        return nodes

    def specify_edges(self, net_params):
        """
        See parent class
        """
        horz_length_in = net_params.additional_params["horizontal_length_in"]
        horz_length_out = net_params.additional_params["horizontal_length_out"]
        vert_length_in = net_params.additional_params["vertical_length_in"]
        vert_length_out = net_params.additional_params["vertical_length_out"]

        edges = [
            {"id": "left", "type": "horizontal", "priority": "78",
             "from": "left", "to": "center", "length": repr(horz_length_in)},
            {"id": "right", "type": "horizontal", "priority": "78",
             "from": "center", "to": "right", "length": repr(horz_length_out)},
            {"id": "bottom", "type": "vertical", "priority": "78",
             "from": "bottom", "to": "center", "length": repr(vert_length_in)},
            {"id": "top", "type": "vertical", "priority": "78",
             "from": "center", "to": "top", "length": repr(vert_length_out)}]

        return edges

    def specify_types(self, net_params):
        """
        See parent class
        """
        horizontal_lanes = net_params.additional_params["horizontal_lanes"]
        vertical_lanes = net_params.additional_params["vertical_lanes"]
        if isinstance(net_params.additional_params["speed_limit"], int) or \
                isinstance(net_params.additional_params["speed_limit"], float):
            speed_limit = {
                "horizontal": net_params.additional_params["speed_limit"],
                "vertical": net_params.additional_params["speed_limit"]}
        else:
            speed_limit = net_params.additional_params["speed_limit"]

        types = [{"id": "horizontal", "numLanes": repr(horizontal_lanes),
                  "speed": repr(speed_limit["horizontal"])},
                 {"id": "vertical", "numLanes": repr(vertical_lanes),
                  "speed": repr(speed_limit["vertical"])}]

        return types

    def specify_routes(self, net_params):
        """
        See parent class
        """
        rts = {"left": ["left", "right"], "bottom": ["bottom", "top"]}
        return rts
