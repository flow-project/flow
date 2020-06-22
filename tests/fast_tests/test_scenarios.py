import unittest
import os
from flow.core.params import VehicleParams
from flow.core.params import NetParams
from flow.networks import BottleneckNetwork, FigureEightNetwork, \
    TrafficLightGridNetwork, HighwayNetwork, RingNetwork, MergeNetwork, \
    MiniCityNetwork, MultiRingNetwork
from flow.networks import I210SubNetwork
from tests.setup_scripts import highway_exp_setup

import flow.config as config

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
                    "num_edges": 1,
                    "use_ghost_edge": False,
                    "ghost_speed_limit": 25,
                    "boundary_cell_length": 300,
                }
            )
        )

    def test_ghost_edge(self):
        """Validate the functionality of the ghost edge feature."""
        # =================================================================== #
        #                         Without a ghost edge                        #
        # =================================================================== #

        # create the network
        env, _, _ = highway_exp_setup(
            net_params=NetParams(additional_params={
                "length": 1000,
                "lanes": 4,
                "speed_limit": 30,
                "num_edges": 1,
                "use_ghost_edge": False,
                "ghost_speed_limit": 25,
                "boundary_cell_length": 300,
            })
        )
        env.reset()

        # check the network length
        self.assertEqual(env.k.network.length(), 1000)

        # check the edge list
        self.assertEqual(env.k.network.get_edge_list(), ["highway_0"])

        # check the speed limits of the edges
        self.assertEqual(env.k.network.speed_limit("highway_0"), 30)

        # =================================================================== #
        #                   With a ghost edge (300m, 25m/s)                   #
        # =================================================================== #

        # create the network
        env, _, _ = highway_exp_setup(
            net_params=NetParams(additional_params={
                "length": 1000,
                "lanes": 4,
                "speed_limit": 30,
                "num_edges": 1,
                "use_ghost_edge": True,
                "ghost_speed_limit": 25,
                "boundary_cell_length": 300,
            })
        )
        env.reset()

        # check the network length
        self.assertEqual(env.k.network.length(), 1300.1)

        # check the edge list
        self.assertEqual(env.k.network.get_edge_list(),
                         ["highway_0", "highway_end"])

        # check the speed limits of the edges
        self.assertEqual(env.k.network.speed_limit("highway_0"), 30)
        self.assertEqual(env.k.network.speed_limit("highway_end"), 25)

        # =================================================================== #
        #                   With a ghost edge (500m, 10m/s)                   #
        # =================================================================== #

        # create the network
        env, _, _ = highway_exp_setup(
            net_params=NetParams(additional_params={
                "length": 1000,
                "lanes": 4,
                "speed_limit": 30,
                "num_edges": 1,
                "use_ghost_edge": True,
                "ghost_speed_limit": 10,
                "boundary_cell_length": 500,
            })
        )
        env.reset()

        # check the network length
        self.assertEqual(env.k.network.length(), 1500.1)

        # check the edge list
        self.assertEqual(env.k.network.get_edge_list(),
                         ["highway_0", "highway_end"])

        # check the speed limits of the edges
        self.assertEqual(env.k.network.speed_limit("highway_0"), 30)
        self.assertEqual(env.k.network.speed_limit("highway_end"), 10)


class TestRingNetwork(unittest.TestCase):

    """Tests RingNetwork in flow/networks/ring.py."""

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

    """Tests MultiRingNetwork in flow/networks/multi_ring.py."""

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


class TestI210SubNetwork(unittest.TestCase):

    """Tests I210SubNetwork in flow/networks/i210_subnetwork.py."""

    def test_additional_net_params(self):
        """Ensures that not returning the correct params leads to an error."""
        self.assertTrue(
            test_additional_params(
                network_class=I210SubNetwork,
                additional_params={
                    "on_ramp": False,
                    "ghost_edge": False,
                }
            )
        )

    def test_specify_routes(self):
        """Validates that the routes are properly specified for the network.

        This is done simply by checking the initial edges routes are specified
        from, which alternates based on choice of network configuration.

        This method tests the routes for the following cases:

        1. on_ramp = False, ghost_edge = False
        2. on_ramp = True,  ghost_edge = False
        3. on_ramp = False, ghost_edge = True
        4. on_ramp = True,  ghost_edge = True
        """
        # test case 1
        network = I210SubNetwork(
            name='test-3',
            vehicles=VehicleParams(),
            net_params=NetParams(
                template=os.path.join(
                    config.PROJECT_PATH,
                    "examples/exp_configs/templates/sumo/test2.net.xml"
                ),
                additional_params={
                    "on_ramp": False,
                    "ghost_edge": False,
                },
            ),
        )

        self.assertEqual(
            ['119257914'],
            sorted(list(network.specify_routes(network.net_params).keys()))
        )

        del network

        # test case 2
        network = I210SubNetwork(
            name='test-3',
            vehicles=VehicleParams(),
            net_params=NetParams(
                template=os.path.join(
                    config.PROJECT_PATH,
                    "examples/exp_configs/templates/sumo/test2.net.xml"
                ),
                additional_params={
                    "on_ramp": True,
                    "ghost_edge": True,
                },
            ),
        )

        self.assertEqual(
            ['119257908#0',
             '119257908#1',
             '119257908#1-AddedOffRampEdge',
             '119257908#1-AddedOnRampEdge',
             '119257908#2',
             '119257908#3',
             '119257914',
             '173381935',
             '27414342#0',
             '27414342#1-AddedOnRampEdge',
             '27414345',
             'ghost0'],
            sorted(list(network.specify_routes(network.net_params).keys()))
        )

        del network

        # test case 3
        network = I210SubNetwork(
            name='test-3',
            vehicles=VehicleParams(),
            net_params=NetParams(
                template=os.path.join(
                    config.PROJECT_PATH,
                    "examples/exp_configs/templates/sumo/test2.net.xml"
                ),
                additional_params={
                    "on_ramp": False,
                    "ghost_edge": True,
                },
            ),
        )

        self.assertEqual(
            ['119257914', 'ghost0'],
            sorted(list(network.specify_routes(network.net_params).keys()))
        )

        del network

        # test case 4
        network = I210SubNetwork(
            name='test-3',
            vehicles=VehicleParams(),
            net_params=NetParams(
                template=os.path.join(
                    config.PROJECT_PATH,
                    "examples/exp_configs/templates/sumo/test2.net.xml"
                ),
                additional_params={
                    "on_ramp": True,
                    "ghost_edge": True,
                },
            ),
        )

        self.assertEqual(
            ['119257908#0',
             '119257908#1',
             '119257908#1-AddedOffRampEdge',
             '119257908#1-AddedOnRampEdge',
             '119257908#2',
             '119257908#3',
             '119257914',
             '173381935',
             '27414342#0',
             '27414342#1-AddedOnRampEdge',
             '27414345',
             'ghost0'],
            sorted(list(network.specify_routes(network.net_params).keys()))
        )

        del network


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
