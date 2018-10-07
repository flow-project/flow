"""Contains the highway scenario class."""

from flow.scenarios.base_scenario import Scenario
from flow.core.params import InitialConfig
from flow.core.traffic_lights import TrafficLights

ADDITIONAL_NET_PARAMS = {
    # length of the highway
    "length": 1000,
    # number of lanes
    "lanes": 4,
    # speed limit for all edges
    "speed_limit": 30,
    # number of edges to divide the highway into
    "num_edges": 1
}


class HighwayScenario(Scenario):
    """Highway scenario class."""

    def __init__(self,
                 name,
                 generator_class,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLights()):
        """Initialize a highway scenario.

        Requires from net_params:
        - length: length of the highway
        - lanes: number of lanes in the highway
        - speed_limit: max speed limit of the highway

        See flow/scenarios/base_scenario.py for description of params.
        """
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        self.length = net_params.additional_params["length"]
        self.lanes = net_params.additional_params["lanes"]
        self.num_edges = net_params.additional_params.get("num_edges", 1)

        super().__init__(name, generator_class, vehicles, net_params,
                         initial_config, traffic_lights)

    def specify_edge_starts(self):
        """See parent class."""
        edgestarts = [("highway_{}".format(i), 0)
                      for i in range(self.num_edges)]
        return edgestarts

    def gen_custom_start_pos(self, initial_config, num_vehicles, **kwargs):
        """Generate a user defined set of starting positions.
        This method is just used for testing.

        Parameters
        ----------
        initial_config : InitialConfig type
            see flow/core/params.py
        num_vehicles : int
            number of vehicles to be placed on the network
        kwargs : dict
            extra components, usually defined during reset to overwrite initial
            config parameters

        Returns
        -------
        startpositions : list of tuple (float, float)
            list of start positions [(edge0, pos0), (edge1, pos1), ...]
        startlanes : list of int
            list of start lanes
        startvel : list of float
            list of start speeds
        """
        # all vehicles start with an initial speed of 0 m/s
        startvel = [0 for _ in range(len(kwargs["start_lanes"]))]

        return kwargs["start_positions"], kwargs["start_lanes"], startvel
