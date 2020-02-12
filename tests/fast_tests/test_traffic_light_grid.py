import unittest

from flow.core.experiment import Experiment

from tests.setup_scripts import traffic_light_grid_mxn_exp_setup


class Test1x1Environment(unittest.TestCase):
    def setUp(self):
        # create the environment and network classes for a traffic light grid network

        self.env, _, flow_params = traffic_light_grid_mxn_exp_setup()
        self.env.reset()

        # instantiate an experiment class
        self.exp = Experiment(flow_params)

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
        edges = [
            "left0_0", "right0_0", "bot0_0", "top0_0", "bot0_1", "top0_1",
            "left1_0", "right1_0", ":center0"
        ]
        for i in range(len(edges)):
            edge = edges[i]
            self.assertEqual(self.env._split_edge(edge), i + 1)

    def test_convert_edge(self):
        edges = [
            "left0_0", "right0_0", "bot0_0", "top0_0", "bot0_1", "top0_1",
            "left1_0", "right1_0", ":center0"
        ]
        self.assertEqual(
            sorted(self.env._convert_edge(edges)),
            [i + 1 for i in range(len(edges))])

    @staticmethod
    def gen_edges(col_num, row_num):
        edges = []
        for i in range(col_num):
            edges += ["left" + str(row_num) + '_' + str(i)]
            edges += ["right" + '0' + '_' + str(i)]

        # build the left and then the right edges
        for i in range(row_num):
            edges += ["bot" + str(i) + '_' + '0']
            edges += ["top" + str(i) + '_' + str(col_num)]

        return edges

    def test_get_distance_to_intersection(self):
        # Obtain list of lists of vehicles on entrance
        # edges, then the distances.
        veh_ids = [
            self.env.k.vehicle.get_ids_by_edge(e) for e in self.gen_edges(1, 1)
        ]
        dists = [self.env.get_distance_to_intersection(v) for v in veh_ids]
        grid = self.env.net_params.additional_params['grid_array']
        short_length = grid['short_length']

        # The first check asserts all the lists are equal. With the default
        # initial config (sans noise) it should be. The second check asserts
        # that all the vehicles are in the confines of [0, short_length] away
        # from the intersection.
        for d_list in dists:
            self.assertListEqual(d_list, dists[0])
            for d in d_list:
                self.assertLessEqual(d, short_length)
                self.assertGreaterEqual(d, 0)

        # Asserts that when a vehicles is in a junction,
        # get_distance_to_intersection returns 0.
        veh_edges = self.env.k.vehicle.get_edge(self.env.k.vehicle.get_ids())
        while not ['center' in edge for edge in veh_edges]:
            print(self.env.k.vehicle.get_edge(self.env.k.vehicle.get_ids()))
            self.env.step(rl_actions=[])
        junction_veh = list(
            filter(lambda x: 'center' in x, self.env.k.vehicle.get_ids()))
        for veh_id in junction_veh:
            self.assertEqual(0, self.env.get_distance_to_intersection(veh_id))


class Test2x2Environment(unittest.TestCase):
    def setUp(self):
        # create the environment and network classes for a traffic light grid network

        self.env, _, flow_params = traffic_light_grid_mxn_exp_setup(row_num=2, col_num=2)
        self.env.reset()

        # instantiate an experiment class
        self.exp = Experiment(flow_params)

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free up used memory
        self.env = None
        self.exp = None

    def test_get_relative_node(self):
        node = self.env._get_relative_node('center0', 'top')
        self.assertEqual(2, node)

        node = self.env._get_relative_node('center0', 'right')
        self.assertEqual(1, node)

        node = self.env._get_relative_node('center0', 'bottom')
        self.assertEqual(-1, node)

        with self.assertRaises(NotImplementedError):
            self.env._get_relative_node('center0', 'blah')

        node = self.env._get_relative_node('center2', 'bottom')
        self.assertEqual(0, node)

        node = self.env._get_relative_node('center2', 'right')
        self.assertEqual(3, node)

        node = self.env._get_relative_node('center2', 'left')
        self.assertEqual(-1, node)

        with self.assertRaises(NotImplementedError):
            self.env._get_relative_node('center1', 'blah')


if __name__ == '__main__':
    unittest.main()
