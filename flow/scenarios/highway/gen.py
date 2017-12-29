from flow.core.generator import Generator

from numpy import pi, sin, cos, linspace


class HighwayGenerator(Generator):
    """
    Generator for loop circle used in MIT traffic simulation. Requires from
    net_params:
     - length: length of the circle
     - lanes: number of lanes in the circle
     - speed_limit: max speed limit of the circle
    """

    def __init__(self, net_params, base):
        """
        See parent class
        """
        length = net_params.additional_params["length"]
        lanes = net_params.additional_params["lanes"]
        self.name = "%s-%dm%dl" % (base, length, lanes)

        super().__init__(net_params, base)

    def specify_nodes(self, net_params):
        """
        See parent class
        """
        length = net_params.additional_params["length"]

        nodes = [{"id": "begin", "x": repr(0),      "y": repr(0)},
                 {"id": "end",   "x": repr(length), "y": repr(0)}]

        return nodes

    def specify_edges(self, net_params):
        """
        See parent class
        """
        length = net_params.additional_params["length"]

        edges = [
            {"id": "highway",
             "type": "highwayType",
             "from": "begin",
             "to": "end",
             "length": repr(length)}
        ]

        return edges

    def specify_types(self, net_params):
        """
        See parent class
        """
        lanes = net_params.additional_params["lanes"]
        speed_limit = net_params.additional_params["speed_limit"]

        types = [{"id": "highwayType",
                  "numLanes": repr(lanes),
                  "speed": repr(speed_limit)}]

        return types

    def specify_routes(self, net_params):
        """
        See parent class
        """
        rts = {"highway": ["highway"]}

        return rts
