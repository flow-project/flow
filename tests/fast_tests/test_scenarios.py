import unittest
import os
from flow.core.params import VehicleParams
from flow.core.params import NetParams
from flow.scenarios import BottleneckScenario, FigureEightScenario, \
    TrafficLightGridScenario, HighwayScenario, RingScenario, MergeScenario, \
    MiniCityScenario, MultiRingScenario

__all__ = [
    "MultiRingScenario", "MiniCityScenario"
]

os.environ["TEST_FLAG"] = "True"


class TestBottleneckScenario(unittest.TestCase):

    """Tests BottleneckScenario in flow/scenarios/bottleneck.py."""

    def test_additional_net_params(self):
        """Ensures that not returning the correct params leads to an error."""
        self.assertTrue(
            test_additional_params(
                scenario_class=BottleneckScenario,
                additional_params={
                    "scaling": 1,
                    'speed_limit': 23
                }
            )
        )


class TestFigureEightScenario(unittest.TestCase):

    """Tests FigureEightScenario in flow/scenarios/figure_eight.py."""

    def test_additional_net_params(self):
        """Ensures that not returning the correct params leads to an error."""
        self.assertTrue(
            test_additional_params(
                scenario_class=FigureEightScenario,
                additional_params={
                    "radius_ring": 30,
                    "lanes": 1,
                    "speed_limit": 30,
                    "resolution": 40
                }
            )
        )


class TestTrafficLightGridScenario(unittest.TestCase):

    """Tests TrafficLightGridScenario in flow/scenarios/traffic_light_grid.py."""

    def test_additional_net_params(self):
        """Ensures that not returning the correct params leads to an error."""
        self.assertTrue(
            test_additional_params(
                scenario_class=TrafficLightGridScenario,
                additional_params={
                    "grid_array": {
                        "row_num": 3,
                        "col_num": 2,
                        "inner_length": None,
                        "short_length": None,
                        "long_length": None,
                        "cars_top": 20,
                        "cars_bot": 20,
                        "cars_left": 20,
                        "cars_right": 20,
                    },
                    "horizontal_lanes": 1,
                    "vertical_lanes": 1,
                    "speed_limit": {
                        "vertical": 35,
                        "horizontal": 35
                    }
                }
            )
        )


class TestHighwayScenario(unittest.TestCase):

    """Tests HighwayScenario in flow/scenarios/highway.py."""

    def test_additional_net_params(self):
        """Ensures that not returning the correct params leads to an error."""
        self.assertTrue(
            test_additional_params(
                scenario_class=HighwayScenario,
                additional_params={
                    "length": 1000,
                    "lanes": 4,
                    "speed_limit": 30,
                    "num_edges": 1
                }
            )
        )


class TestRingScenario(unittest.TestCase):

    """Tests RingScenario in flow/scenarios/ring.py."""

    def test_additional_net_params(self):
        """Ensures that not returning the correct params leads to an error."""
        self.assertTrue(
            test_additional_params(
                scenario_class=RingScenario,
                additional_params={
                    "length": 230,
                    "lanes": 1,
                    "speed_limit": 30,
                    "resolution": 40
                }
            )
        )


class TestMergeScenario(unittest.TestCase):

    """Tests MergeScenario in flow/scenarios/merge.py."""

    def test_additional_net_params(self):
        """Ensures that not returning the correct params leads to an error."""
        self.assertTrue(
            test_additional_params(
                scenario_class=MergeScenario,
                additional_params={
                    "merge_length": 100,
                    "pre_merge_length": 200,
                    "post_merge_length": 100,
                    "merge_lanes": 1,
                    "highway_lanes": 1,
                    "speed_limit": 30
                }
            )
        )


class TestMultiRingScenario(unittest.TestCase):

    """Tests MultiRingScenario in flow/scenarios/multi_ring.py."""

    def test_additional_net_params(self):
        """Ensures that not returning the correct params leads to an error."""
        self.assertTrue(
            test_additional_params(
                scenario_class=MultiRingScenario,
                additional_params={
                    "length": 230,
                    "lanes": 1,
                    "speed_limit": 30,
                    "resolution": 40,
                    "num_rings": 7
                }
            )
        )


###############################################################################
#                              Utility methods                                #
###############################################################################


def test_additional_params(scenario_class,
                           additional_params):
    """Test that the environment raises an Error in any param is missing.

    Parameters
    ----------
    scenario_class : flow.scenarios.*
        the scenario class that this method will try to instantiate
    additional_params : dict
        the valid and required additional parameters for the environment in
        NetParams

    Returns
    -------
    bool
        True if the test passed, False otherwise
    """
    for key in additional_params.keys():
        # remove one param from the additional_params dict
        new_add = additional_params.copy()
        del new_add[key]

        try:
            scenario_class(
                name='test',
                vehicles=VehicleParams(),
                net_params=NetParams(additional_params=new_add)
            )
            # if no KeyError is raised, the test has failed, so return False
            return False
        except KeyError:
            # if a KeyError is raised, test the next param
            pass

    # if removing all additional params led to KeyErrors, the test has passed,
    # so return True
    return True


###############################################################################
#                                End of utils                                 #
###############################################################################


if __name__ == '__main__':
    unittest.main()
