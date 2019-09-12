"""Contains the highway with ramps network class."""

from flow.networks.base import Network
from flow.core.params import InitialConfig, TrafficLightParams
from collections import defaultdict
from numpy import pi, sin, cos


ADDITIONAL_NET_PARAMS = {
    # lengths of highway, on-ramps and off-ramps respectively
    "highway_length": 300,
    "on_ramps_length": 100,
    "off_ramps_length": 100,
    # number of lanes on highway, on-ramps and off-ramps respectively
    "highway_lanes": 1,
    "on_ramps_lanes": 1,
    "off_ramps_lanes": 1,
    # speed limit on highway, on-ramps and off-ramps respectively
    "highway_speed": 10,
    "on_ramps_speed": 10,
    "off_ramps_speed": 10,
    # positions of the on-ramps
    "on_ramps_pos": [],
    # positions of the off-ramps
    "off_ramps_pos": [],
    # probability for a vehicle to exit the highway at the next off-ramp
    "next_off_ramp_proba": 0.2,
    # ramps angles
    "angle_on_ramps": - 3 * pi / 4,
    "angle_off_ramps": - pi / 4
}


class HighwayRampsNetwork(Network):
    """Network class for a highway section with on and off ramps.

    This network consists of a single or multi-lane highway network with a
    variable number of on-ramps and off-ramps at arbitrary positions,
    with arbitrary numbers of lanes. It can be used to generate periodic
    perturbation on a more realistic highway.

    Parameters in net_params:

    * **highway_length** : total length of the highway
    * **on_ramps_length** : length of each on-ramp
    * **off_ramps_length** : length of each off-ramp
    * **highway_lanes** : number of lanes on the highway
    * **on_ramps_lanes** : number of lanes on each on-ramp
    * **off_ramps_lanes** : number of lanes on each off-ramp
    * **highway_speed** : speed limit on the highway
    * **on_ramps_speed** : speed limit on each on-ramp
    * **off_ramps_speed** : speed limit on each off-ramp
    * **on_ramps_pos** : positions of the in-ramps on the highway (int list)
    * **off_ramps_pos** : positions of the off-ramps on the highway (int list)
    * **next_off_ramp_proba** : probability for a vehicle to exit the highway
                                at the next off-ramp
    """

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):
        """Initialize a highway with on and off ramps network."""
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        # load parameters into class
        params = net_params.additional_params

        self.highway_length = params['highway_length']
        self.on_ramps_length = params['on_ramps_length']
        self.off_ramps_length = params['off_ramps_length']

        self.highway_lanes = params['highway_lanes']
        self.on_ramps_lanes = params['on_ramps_lanes']
        self.off_ramps_lanes = params['off_ramps_lanes']

        self.highway_speed = params['highway_speed']
        self.on_ramps_speed = params['on_ramps_speed']
        self.off_ramps_speed = params['off_ramps_speed']

        self.on_ramps_pos = params['on_ramps_pos']
        self.off_ramps_pos = params['off_ramps_pos']

        self.p = params['next_off_ramp_proba']

        self.angle_on_ramps = params['angle_on_ramps']
        self.angle_off_ramps = params['angle_off_ramps']

        # generate position of all network nodes
        self.ramps_pos = sorted(self.on_ramps_pos + self.off_ramps_pos)
        self.nodes_pos = sorted(list(set([0] + self.ramps_pos +
                                         [self.highway_length])))

        # highway_pos[x] = id of the highway node whose starting position is x
        self.highway_pos = {x: i for i, x in enumerate(self.nodes_pos)}
        # ramp_pos[x] = id of the ramp node whose intersection with the highway
        # is at position x
        self.ramp_pos = {x: "on_ramp_{}".format(i)
                         for i, x in enumerate(self.on_ramps_pos)}
        self.ramp_pos.update({x: "off_ramp_{}".format(i)
                             for i, x in enumerate(self.off_ramps_pos)})

        # make sure network is constructable
        if (len(self.ramps_pos) > 0 and
           (min(self.ramps_pos) <= 0 or
           max(self.ramps_pos) >= self.highway_length)):
            raise ValueError('All ramps positions should be positive and less '
                             'than highway length. Current ramps positions: {}'
                             '. Current highway length: {}.'.format(
                                 self.ramps_pos, self.highway_length))
        if len(self.ramps_pos) != len(list(set(self.ramps_pos))):
            raise ValueError('Two ramps positions cannot be equal.')

        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)

    def specify_nodes(self, net_params):
        """See parent class."""
        nodes_highway = [{
            "id": "highway_{}".format(i),
            "x": self.nodes_pos[i],
            "y": 0,
            "radius": 10
        } for i in range(len(self.nodes_pos))]

        nodes_on_ramps = [{
            "id": "on_ramp_{}".format(i),
            "x": x + self.on_ramps_length * cos(self.angle_on_ramps),
            "y": self.on_ramps_length * sin(self.angle_on_ramps)
        } for i, x in enumerate(self.on_ramps_pos)]

        nodes_off_ramps = [{
            "id": "off_ramp_{}".format(i),
            "x": x + self.off_ramps_length * cos(self.angle_off_ramps),
            "y": self.off_ramps_length * sin(self.angle_off_ramps)
        } for i, x in enumerate(self.off_ramps_pos)]

        return nodes_highway + nodes_on_ramps + nodes_off_ramps

    def specify_edges(self, net_params):
        """See parent class."""
        highway_edges = [{
            "id": "highway_{}".format(i),
            "type": "highway",
            "from": "highway_{}".format(i),
            "to": "highway_{}".format(i + 1),
            "length": self.nodes_pos[i + 1] - self.nodes_pos[i]
        } for i in range(len(self.nodes_pos) - 1)]

        on_ramps_edges = [{
            "id": "on_ramp_{}".format(i),
            "type": "on_ramp",
            "from": "on_ramp_{}".format(i),
            "to": "highway_{}".format(self.highway_pos[x]),
            "length": self.on_ramps_length
        } for i, x in enumerate(self.on_ramps_pos)]

        off_ramps_edges = [{
            "id": "off_ramp_{}".format(i),
            "type": "off_ramp",
            "from": "highway_{}".format(self.highway_pos[x]),
            "to": "off_ramp_{}".format(i),
            "length": self.off_ramps_length
        } for i, x in enumerate(self.off_ramps_pos)]

        return highway_edges + on_ramps_edges + off_ramps_edges

    def specify_routes(self, net_params):
        """See parent class."""
        def get_routes(start_node_pos):
            """Compute the routes recursively."""
            if start_node_pos not in self.nodes_pos:
                raise ValueError('{} is not a node position.'.format(
                    start_node_pos))

            id_highway_node = self.highway_pos[start_node_pos]

            if id_highway_node + 1 >= len(self.nodes_pos):
                return [(["highway_{}".format(id_highway_node - 1)], 1)]

            id_ramp_node = self.ramp_pos[start_node_pos]

            routes = get_routes(self.nodes_pos[id_highway_node + 1])

            if id_ramp_node.startswith("on"):
                return ([
                    (["highway_{}".format(id_highway_node - 1)] + route, prob)
                    for route, prob in routes if not route[0].startswith("on")
                ] + [
                    ([id_ramp_node] + route, prob)
                    for route, prob in routes if not route[0].startswith("on")
                ] + [
                    (route, prob)
                    for route, prob in routes if route[0].startswith("on")
                ])
            else:
                return ([
                    (["highway_{}".format(id_highway_node - 1)] + route,
                     (1 - self.p) * prob)
                    for route, prob in routes if not route[0].startswith("on")
                ] + [
                    (["highway_{}".format(id_highway_node - 1), id_ramp_node],
                     self.p * prob)
                    for route, prob in routes if not route[0].startswith("on")
                ] + [
                    (route, prob)
                    for route, prob in routes if route[0].startswith("on")
                ])

        routes = get_routes(self.nodes_pos[1])

        rts = defaultdict(list)
        for route, prob in routes:
            rts[route[0]].append((route, prob))

        return rts

    def specify_types(self, net_params):
        """See parent class."""
        types = [{
            "id": "highway",
            "numLanes": self.highway_lanes,
            "speed": self.highway_speed
        }, {
            "id": "on_ramp",
            "numLanes": self.on_ramps_lanes,
            "speed": self.on_ramps_speed
        }, {
            "id": "off_ramp",
            "numLanes": self.off_ramps_lanes,
            "speed": self.off_ramps_speed
        }]

        return types
