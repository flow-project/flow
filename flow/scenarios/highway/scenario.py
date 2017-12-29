from flow.scenarios.base_scenario import Scenario
from flow.core.params import InitialConfig


class HighwayScenario(Scenario):
    def __init__(self, name, generator_class, vehicles, net_params,
                 initial_config=InitialConfig()):
        """
        Initializes a loop scenario. Required net_params: length, lanes,
        speed_limit, resolution.

        See Scenario.py for description of params.
        """
        if "length" not in net_params.additional_params:
            raise ValueError("length of highway not supplied")
        self.length = net_params.additional_params["length"]

        if "lanes" not in net_params.additional_params:
            raise ValueError("lanes of highway not supplied")
        self.lanes = net_params.additional_params["lanes"]

        if "speed_limit" not in net_params.additional_params:
            raise ValueError("speed limit of highway not supplied")
        self.speed_limit = net_params.additional_params["speed_limit"]

        super().__init__(name, generator_class, vehicles, net_params,
                         initial_config=initial_config)

    def specify_edge_starts(self):
        """
        See parent class
        """
        edgestarts = [("highway", 0)]

        return edgestarts
