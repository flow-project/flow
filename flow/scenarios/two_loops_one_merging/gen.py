from flow.core.generator import Generator

from numpy import pi, sin, cos, linspace, arcsin


class TwoLoopOneMergingGenerator(Generator):
    """
    Generator for a two-loop network in which both loops merge into a common
    lane. Requires from net_params:
     - ring_radius: radius of the smaller ring road (the larger has 1.5x this
       radius)
     - lanes: number of lanes in the network
     - speed_limit: max speed limit in the network
     - resolution: number of nodes resolution
    """

    def __init__(self, net_params, net_path, cfg_path, base):
        """
        See parent class
        """
        radius = net_params.additional_params["ring_radius"]
        lanes = net_params.additional_params["lanes"]

        super().__init__(net_params, net_path, cfg_path, base)

        self.name = "%s-%dr%dl" % (base, radius, lanes)

    def specify_nodes(self, net_params):
        """
        See parent class
        """
        r = net_params.additional_params["ring_radius"]
        angle_small = pi / 3
        angle_large = arcsin(0.75)

        nodes = [
            {"id": "bottom", "x": repr(0), "y": repr(-r * sin(angle_small))},
            {"id": "right", "x": repr(r * (1 + cos(angle_large))),
             "y": repr(0)},
            {"id": "top", "x": repr(0), "y": repr(r * sin(angle_small))},
            {"id": "left", "x": repr(-r * (1 + cos(angle_small))),
             "y": repr(0)}]

        return nodes

    def specify_edges(self, net_params):
        """
        See parent class
        """
        r = net_params.additional_params["ring_radius"]
        resolution = net_params.additional_params["resolution"]

        angle_small = pi / 3
        angle_large = arcsin(0.75)
        merge_edgelen = (pi - angle_large) * (1.5 * r)
        ring_edgelen = 2 / 3 * pi * r

        edges = [
            {"id": "center",
             "type": "edgeType",
             "from": "bottom",
             "to": "top",
             "length": repr(ring_edgelen),
             "shape": " ".join(
                 ["%.2f,%.2f" % (r * (cos(t) - cos(angle_small)), r * sin(t))
                  for t in linspace(- pi / 3, pi / 3, resolution)])},

            {"id": "right_top",
             "type": "edgeType",
             "from": "right",
             "to": "top",
             "length": repr(merge_edgelen),
             "shape": " ".join(["%.2f,%.2f" % (
             1.5 * r * (cos(t) + cos(angle_large)), 1.5 * r * sin(t))
                                for t in
                                linspace(0, pi - angle_large, resolution)])},

            {"id": "right_bottom",
             "type": "edgeType",
             "from": "bottom",
             "to": "right",
             "length": repr(merge_edgelen),
             "shape": " ".join(["%.2f,%.2f" % (
                 1.5 * r * (cos(t) + cos(angle_large)), 1.5 * r * sin(t))
                                for t in linspace(pi + angle_large, 2 * pi,
                                                  resolution)])},

            {"id": "left_top",
             "type": "edgeType",
             "from": "top",
             "to": "left",
             "length": repr(ring_edgelen),
             "shape": " ".join(["%.2f,%.2f" % (
                 r * (cos(t) - cos(angle_small)), r * sin(t))
                                for t in
                                linspace(pi / 3, pi, resolution)])},

            {"id": "left_bottom",
             "type": "edgeType",
             "from": "left",
             "to": "bottom",
             "length": repr(ring_edgelen),
             "shape": " ".join(["%.2f,%.2f" % (
                 r * (cos(t) - cos(angle_small)), r * sin(t))
                                for t in
                                linspace(pi, 5 / 3 * pi, resolution)])}]

        return edges

    def specify_types(self, net_params):
        """
        See parent class
        """
        lanes = net_params.additional_params["lanes"]
        speed_limit = net_params.additional_params["speed_limit"]
        types = [{"id": "edgeType", "numLanes": repr(lanes),
                  "speed": repr(speed_limit)}]

        return types

    def specify_routes(self, net_params):
        """
        See parent class
        """
        rts = {"right_top": ["right_top", "left_top", "left_bottom",
                             "right_bottom", "right_top"],
               "right_bottom": ["right_bottom", "right_top", "left_top",
                                "left_bottom", "right_bottom"],
               "left_top": ["left_top", "left_bottom", "center", "left_top"],
               "left_bottom": ["left_bottom", "center", "left_top",
                               "left_bottom"],
               "center": ["center", "left_top", "left_bottom"]}

        return rts
