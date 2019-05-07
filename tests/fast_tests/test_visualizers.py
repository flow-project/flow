# from flow.visualize import visualizer_rllab as vs_rllab
# from flow.visualize.visualizer_rllab import visualizer_rllab
from flow.visualize import visualizer_rllib as vs_rllib
from flow.visualize.visualizer_rllib import visualizer_rllib
import flow.visualize.capacity_diagram_generator as cdg
import flow.visualize.time_space_diagram as tsd

import os
import unittest
import ray
import numpy as np

os.environ['TEST_FLAG'] = 'True'


class TestVisualizerRLlib(unittest.TestCase):
    """Tests visualizer_rllib"""

    def test_visualizer_single(self):
        """Test for single agent"""
        try:
            ray.init(num_cpus=1)
        except Exception:
            pass
        # current path
        current_path = os.path.realpath(__file__).rsplit('/', 1)[0]

        # run the experiment and check it doesn't crash
        arg_str = '{}/../data/rllib_data/single_agent 1 --num-rollouts 1 ' \
                  '--render_mode no_render ' \
                  '--horizon 10'.format(current_path).split()
        parser = vs_rllib.create_parser()
        pass_args = parser.parse_args(arg_str)
        visualizer_rllib(pass_args)

    # FIXME(ev) set the horizon so that this runs faster
    def test_visualizer_multi(self):
        """Test for multi-agent visualization"""
        try:
            ray.init(num_cpus=1)
        except Exception:
            pass
        # current path
        current_path = os.path.realpath(__file__).rsplit('/', 1)[0]

        # run the experiment and check it doesn't crash
        arg_str = '{}/../data/rllib_data/multi_agent 1 --num-rollouts 1 ' \
                  '--render_mode no_render ' \
                  '--horizon 10'.format(current_path).split()
        parser = vs_rllib.create_parser()
        pass_args = parser.parse_args(arg_str)
        visualizer_rllib(pass_args)


# class TestVisualizerRLlab(unittest.TestCase):
#     """Tests visualizer_rllab"""
#
#     def test_visualizer(self):
#         # current path
#         current_path = os.path.realpath(__file__).rsplit('/', 1)[0]
#         arg_str = '{}/../data/rllab_data/itr_0.pkl --num_rollouts 1 ' \
#                   '--no_render'.format(current_path).split()
#         parser = vs_rllab.create_parser()
#         pass_args = parser.parse_args(arg_str)
#         visualizer_rllab(pass_args)


class TestPlotters(unittest.TestCase):

    def test_capacity_diagram_generator(self):
        # import the csv file
        dir_path = os.path.dirname(os.path.realpath(__file__))
        data = cdg.import_data_from_csv(
            os.path.join(dir_path, 'test_files/inflows_outflows.csv'))

        # compute the mean and std of the outflows for all unique inflows
        unique_inflows, mean_outflows, std_outflows = cdg.get_capacity_data(
            data)

        # test that the values match the expected from the
        expected_unique_inflows = np.array([
            400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500,
            1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600,
            2700, 2800, 2900])
        expected_means = np.array([
            385.2, 479.52, 575.28, 668.16, 763.2, 856.8, 900.95668831,
            1029.6705856, 1111.62035833, 1187.87297462, 1258.81962238,
            1257.30378783, 1161.28280975, 1101.85671862, 1261.26596639,
            936.91255623, 1039.90127834, 1032.13903881, 937.70410361,
            934.85669105, 837.58808324, 889.17167643, 892.78528048,
            937.85757297, 934.86027655, 804.14440138])
        expected_stds = np.array([
            1.60996894, 1.44, 1.44, 2.38796985, 2.78854801, 3.6, 149.57165793,
            37.82554569, 67.35786443, 135.35337939, 124.41794128, 221.64466355,
            280.88707947, 199.2875712, 258.72510896, 194.0785382, 239.71034056,
            182.75627664, 331.37899239, 325.82943015, 467.54641633,
            282.15049541, 310.36329236, 92.61828854, 229.6155371,
            201.29461492])

        np.testing.assert_array_almost_equal(unique_inflows,
                                             expected_unique_inflows)
        np.testing.assert_array_almost_equal(mean_outflows, expected_means)
        np.testing.assert_array_almost_equal(std_outflows, expected_stds)

    def test_time_space_diagram_figure_eight(self):
        # check that the exported data matches the expected emission file data
        fig8_emission_data = {
            'idm_3': {'pos': [27.25, 28.25, 30.22, 33.17],
                      'time': [1.0, 2.0, 3.0, 4.0],
                      'vel': [0.0, 0.99, 1.98, 2.95],
                      'edge': ['upper_ring', 'upper_ring', 'upper_ring',
                               'upper_ring']},
            'idm_4': {'pos': [56.02, 57.01, 58.99, 61.93],
                      'time': [1.0, 2.0, 3.0, 4.0],
                      'vel': [0.0, 0.99, 1.98, 2.95],
                      'edge': ['upper_ring', 'upper_ring', 'upper_ring',
                               'upper_ring']},
            'idm_5': {'pos': [84.79, 85.78, 87.76, 90.7],
                      'time': [1.0, 2.0, 3.0, 4.0],
                      'vel': [0.0, 0.99, 1.98, 2.95],
                      'edge': ['upper_ring', 'upper_ring', 'upper_ring',
                               'upper_ring']},
            'idm_2': {'pos': [28.77, 29.76, 1.63, 4.58],
                      'time': [1.0, 2.0, 3.0, 4.0],
                      'vel': [0.0, 0.99, 1.97, 2.95],
                      'edge': ['top', 'top', 'upper_ring', 'upper_ring']},
            'idm_13': {'pos': [106.79, 107.79, 109.77, 112.74],
                       'time': [1.0, 2.0, 3.0, 4.0],
                       'vel': [0.0, 0.99, 1.98, 2.96],
                       'edge': ['lower_ring', 'lower_ring', 'lower_ring',
                                'lower_ring']},
            'idm_9': {'pos': [22.01, 23.0, 24.97, 27.92],
                      'time': [1.0, 2.0, 3.0, 4.0],
                      'vel': [0.0, 0.99, 1.97, 2.95],
                      'edge': ['left', 'left', 'left', 'left']},
            'idm_6': {'pos': [113.56, 114.55, 116.52, 119.47],
                      'time': [1.0, 2.0, 3.0, 4.0],
                      'vel': [0.0, 0.99, 1.97, 2.95],
                      'edge': ['upper_ring', 'upper_ring', 'upper_ring',
                               'upper_ring']},
            'idm_8': {'pos': [29.44, 0.28, 2.03, 4.78],
                      'time': [1.0, 2.0, 3.0, 4.0],
                      'vel': [0.0, 0.84, 1.76, 2.75],
                      'edge': ['right', ':center_0', ':center_0',
                               ':center_0']},
            'idm_12': {'pos': [78.03, 79.02, 80.99, 83.94],
                       'time': [1.0, 2.0, 3.0, 4.0],
                       'vel': [0.0, 0.99, 1.98, 2.95],
                       'edge': ['lower_ring', 'lower_ring', 'lower_ring',
                                'lower_ring']},
            'idm_10': {'pos': [20.49, 21.48, 23.46, 26.41],
                       'time': [1.0, 2.0, 3.0, 4.0],
                       'vel': [0.0, 0.99, 1.98, 2.95],
                       'edge': ['lower_ring', 'lower_ring', 'lower_ring',
                                'lower_ring']},
            'idm_11': {'pos': [49.26, 50.25, 52.23, 55.17],
                       'time': [1.0, 2.0, 3.0, 4.0],
                       'vel': [0.0, 0.99, 1.98, 2.95],
                       'edge': ['lower_ring', 'lower_ring', 'lower_ring',
                                'lower_ring']},
            'idm_1': {'pos': [0.0, 0.99, 2.97, 5.91],
                      'time': [1.0, 2.0, 3.0, 4.0],
                      'vel': [0.0, 0.99, 1.98, 2.95],
                      'edge': ['top', 'top', 'top', 'top']},
            'idm_7': {'pos': [0.67, 1.66, 3.64, 6.58],
                      'time': [1.0, 2.0, 3.0, 4.0],
                      'vel': [0.0, 0.99, 1.97, 2.94],
                      'edge': ['right', 'right', 'right', 'right']},
            'idm_0': {'pos': [0.0, 1.0, 2.98, 5.95],
                      'time': [1.0, 2.0, 3.0, 4.0],
                      'vel': [0.0, 1.0, 1.99, 2.97],
                      'edge': ['bottom', 'bottom', 'bottom', 'bottom']}
        }
        dir_path = os.path.dirname(os.path.realpath(__file__))
        actual_emission_data = tsd.import_data_from_emission(
            os.path.join(dir_path, 'test_files/fig8_emission.csv'))
        self.assertDictEqual(fig8_emission_data, actual_emission_data)

    def test_time_space_diagram_merge(self):
        pass

    def test_time_space_diagram_ring_road(self):
        pass


if __name__ == '__main__':
    ray.init(num_cpus=1)
    unittest.main()
    ray.shutdown()
