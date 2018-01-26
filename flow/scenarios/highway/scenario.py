from flow.scenarios.base_scenario import Scenario
from flow.core.params import InitialConfig
from flow.core.traffic_lights import TrafficLights


class HighwayScenario(Scenario):
    def __init__(self, name, generator_class, vehicles, net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLights()):
        """
        Initializes a loop scenario. Required net_params: length, lanes,
        speed_limit, resolution.

        See Scenario.py for description of params.
        """
        REQUIRED_NET_PARAMS = ["length", "lanes", "speed_limit"]

        for param in REQUIRED_NET_PARAMS:
            if param not in net_params.additional_params:
                raise ValueError(
                    "highway network parameter {} not supplied".format(param))

        self.length = net_params.additional_params["length"]
        self.lanes = net_params.additional_params["lanes"]
        self.speed_limit = net_params.additional_params["speed_limit"]

        super().__init__(name, generator_class, vehicles, net_params,
                         initial_config, traffic_lights)

    def specify_edge_starts(self):
        """
        See parent class
        """
        edgestarts = [("highway", 0)]

        return edgestarts
