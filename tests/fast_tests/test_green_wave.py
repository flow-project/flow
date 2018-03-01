import unittest

from flow.core.experiment import SumoExperiment

from tests.setup_scripts import grid_mxn_exp_setup


class TestEnvironment(unittest.TestCase):
    def setUp(self):
        # create the environment and scenario classes for a ring road
        self.env, self.scenario = grid_mxn_exp_setup()

        # instantiate an experiment class
        self.exp = SumoExperiment(self.env, self.scenario)

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free up used memory
        self.env = None
        self.exp = None

    def test_split_edge(self):
        """
        In a 1x1 grid, edges are:
        [left0_0, right0_0, bot0_0, top0_0, bot0_1, top0_1,
        left1_0, right1_0, :center0] and should be indexed as such
        """
        edges = ["left0_0", "right0_0", "bot0_0", "top0_0", "bot0_1", "top0_1",
                 "left1_0", "right1_0", ":center0"]
        for i in range(len(edges)):
            edge = edges[i]
            self.assertEqual(self.env._split_edge(edge), i + 1)

    def test_convert_edge(self):
        edges = ["left0_0", "right0_0", "bot0_0", "top0_0", "bot0_1", "top0_1",
                 "left1_0", "right1_0", ":center0"]
        self.assertEqual(sorted(self.env._convert_edge(edges)),
                         [i + 1 for i in range(len(edges))])


if __name__ == '__main__':
    unittest.main()
