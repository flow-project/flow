from flow.scenarios.base_scenario import Scenario


class TwoLaneMergeScenario(Scenario):
    """
    Scenario class for Bay Bridge toll simulations. Like it's generator, it
    is static.
    """
    def specify_edge_starts(self):
        """
        See parent class
        """
        return [("1", 0),
                ("2", 200)]
