from flow.scenarios.base_scenario import Scenario
from flow.core.params import InitialConfig
from flow.core.traffic_lights import TrafficLights


ADDITIONAL_NET_PARAMS = {
    # length of the highway
    "length": 230,
    # number of lanes
    "lanes": 1,
    # speed limit for all edges
    "speed_limit": 30,
}


class HighwayScenario(Scenario):
    def __init__(self, name, generator_class, vehicles, net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLights()):
        """Initializes a highway scenario.

        Requires from net_params:
        - length: length of the circle
        - lanes: number of lanes in the circle
        - speed_limit: max speed limit of the circle

        See Scenario.py for description of params.
        """
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network param "{}" not supplied'.format(p))

        self.length = net_params.additional_params["length"]
        self.lanes = net_params.additional_params["lanes"]

        super().__init__(name, generator_class, vehicles, net_params,
                         initial_config, traffic_lights)

    def specify_edge_starts(self):
        """
        See parent class
        """
        edgestarts = [("highway", 0)]

        return edgestarts
