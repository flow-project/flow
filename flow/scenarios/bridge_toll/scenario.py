from flow.scenarios.base_scenario import Scenario


class BBTollScenario(Scenario):
    """
    Scenario class for Bay Bridge toll simulations. Like it's generator, it
    is static.
    """
    def specify_edge_starts(self):
        """
        See parent class
        """
        return [("1", 0),
                ("2", 100),
                ("3", 405),
                ("4", 425),
                ("5", 580)]

    def get_bottleneck_lanes(self, lane):
        return [int(lane/2), int(lane/4)]

