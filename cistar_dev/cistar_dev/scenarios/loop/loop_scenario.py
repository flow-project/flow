from cistar_dev.core.scenario import Scenario


class LoopScenario(Scenario):
    def __init__(self, name, generator_class, type_params, net_params, cfg_params=None,
                 initial_config=None):
        """
        Initializes a loop scenario. Required net_params: length, lanes,
        speed_limit, resolution. Required initial_config: positions.

        See Scenario.py for description of params.
        """
        if "length" not in net_params:
            raise ValueError("length of circle not supplied")
        self.length = net_params["length"]

        if "lanes" not in net_params:
            raise ValueError("lanes of circle not supplied")
        self.lanes = net_params["lanes"]

        if "speed_limit" not in net_params:
            raise ValueError("speed limit of circle not supplied")
        self.speed_limit = net_params["speed_limit"]

        if "resolution" not in net_params:
            raise ValueError("resolution of circle not supplied")
        self.resolution = net_params["resolution"]

        super().__init__(name, generator_class, type_params, net_params, cfg_params=cfg_params,
                         initial_config=initial_config)

    def specify_edge_starts(self):
        """
        See parent class
        """
        edgelen = self.length / 4

        edgestarts = [("bottom", 0), ("right", edgelen),
                      ("top", 2 * edgelen), ("left", 3 * edgelen)]

        return edgestarts
