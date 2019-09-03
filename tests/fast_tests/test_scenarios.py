import unittest
import os
from flow.core.params import VehicleParams
from flow.core.params import NetParams
from flow.networks import BottleneckNetwork, FigureEightNetwork, \
    TrafficLightGridNetwork, HighwayNetwork, RingNetwork, MergeNetwork, \
    MiniCityNetwork, MultiRingNetwork

__all__ = [
    "MultiRingNetwork", "MiniCityNetwork"
]

os.environ["TEST_FLAG"] = "True"


class TestBottleneckNetwork(unittest.TestCase):

    """Tests BottleneckNetwork in flow/networks/bottleneck.py."""

    def test_additional_net_params(self):
        """Ensures that not returning the correct params leads to an error."""
        self.assertTrue(
            test_additional_params(
                network_class=BottleneckNetwork,
                additional_params={
                    "scaling": 1,
                    'speed_limit': 23
                }
            )
        )


class TestFigureEightNetwork(unittest.TestCase):

    """Tests FigureEightNetwork in flow/networks/figure_eight.py."""

    def test_additional_net_params(self):
        """Ensures that not returning the correct params leads to an error."""
        self.assertTrue(
            test_additional_params(
                network_class=FigureEightNetwork,
                additional_params={
                    "radius_ring": 30,
                    "lanes": 1,
                    "speed_limit": 30,
                    "resolution": 40
                }
            )
        )


class TestTrafficLightGridNetwork(unittest.TestCase):

    """Tests TrafficLightGridNetwork in flow/networks/traffic_light_grid.py."""

    def test_additional_net_params(self):
        """Ensures that not returning the correct params leads to an error."""
        self.assertTrue(
            test_additional_params(
                network_class=TrafficLightGridNetwork,
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


class TestHighwayNetwork(unittest.TestCase):

    """Tests HighwayNetwork in flow/networks/highway.py."""

    def test_additional_net_params(self):
        """Ensures that not returning the correct params leads to an error."""
        self.assertTrue(
            test_additional_params(
                network_class=HighwayNetwork,
                additional_params={
                    "length": 1000,
                    "lanes": 4,
                    "speed_limit": 30,
                    "num_edges": 1
                }
            )
        )


class TestRingNetwork(unittest.TestCase):

    """Tests LoopNetwork in flow/networks/ring.py."""

    def test_additional_net_params(self):
        """Ensures that not returning the correct params leads to an error."""
        self.assertTrue(
            test_additional_params(
                network_class=RingNetwork,
                additional_params={
                    "length": 230,
                    "lanes": 1,
                    "speed_limit": 30,
                    "resolution": 40
                }
            )
        )


class TestMergeNetwork(unittest.TestCase):

    """Tests MergeNetwork in flow/networks/merge.py."""

    def test_additional_net_params(self):
        """Ensures that not returning the correct params leads to an error."""
        self.assertTrue(
            test_additional_params(
                network_class=MergeNetwork,
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


class TestMultiRingNetwork(unittest.TestCase):

    """Tests MultiLoopNetwork in flow/networks/multi_ring.py."""

    def test_additional_net_params(self):
        """Ensures that not returning the correct params leads to an error."""
        self.assertTrue(
            test_additional_params(
                network_class=MultiRingNetwork,
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


def test_additional_params(network_class,
                           additional_params):
    """Test that the environment raises an Error in any param is missing.

    Parameters
    ----------
    network_class : flow.networks.*
        the network class that this method will try to instantiate
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
            network_class(
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
