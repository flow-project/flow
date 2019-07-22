# from flow.visualize import visualizer_rllab as vs_rllab
# from flow.visualize.visualizer_rllab import visualizer_rllab
from flow.visualize import visualizer_rllib as vs_rllib
from flow.visualize.visualizer_rllib import visualizer_rllib
import flow.visualize.capacity_diagram_generator as cdg
import flow.visualize.time_space_diagram as tsd
import flow.visualize.plot_ray_results as prr

import os
import unittest
import ray
import numpy as np
import contextlib
from io import StringIO

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
        arg_str = '{}/../data/rllib_data/single_agent 1 --num_rollouts 1 ' \
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
        arg_str = '{}/../data/rllib_data/multi_agent 1 --num_rollouts 1 ' \
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

        # test get_time_space_data for figure eight networks
        flow_params = tsd.get_flow_params(
            os.path.join(dir_path, 'test_files/fig8.json'))
        pos, speed, _ = tsd.get_time_space_data(
            actual_emission_data, flow_params)

        expected_pos = np.array(
            [[60, 23.8, 182.84166941, 154.07166941, 125.30166941, 96.54166941,
              -203.16166941, -174.40166941, -145.63166941, -116.86166941,
              -88.09166941, -59.33, -30.56, -1.79],
             [59, 22.81, 181.85166941, 153.08166941, 124.31166941, 95.54166941,
              -202.17166941, -173.40166941, -144.64166941, -115.87166941,
              -87.10166941, -58.34, -29.72, -0.8],
             [57.02, 20.83, 179.87166941, 151.10166941, 122.34166941,
              93.56166941, -200.02166941, -171.43166941, -142.66166941,
              -113.89166941, -85.13166941, -56.36, -27.97, 208.64166941]]
        )
        expected_speed = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99,
             0.99, 0.84, 0.99],
            [1.99, 1.98, 1.98, 1.98, 1.98, 1.98, 1.97, 1.98, 1.98, 1.98, 1.97,
             1.97, 1.76, 1.97]
        ])

        np.testing.assert_array_almost_equal(pos[:-1, :], expected_pos)
        np.testing.assert_array_almost_equal(speed[:-1, :], expected_speed)

    def test_time_space_diagram_merge(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        emission_data = tsd.import_data_from_emission(
            os.path.join(dir_path, 'test_files/merge_emission.csv'))

        flow_params = tsd.get_flow_params(
            os.path.join(dir_path, 'test_files/merge.json'))
        pos, speed, _ = tsd.get_time_space_data(emission_data, flow_params)

        expected_pos = np.array(
            [[4.86, 180.32, 361.32, 547.77, 0],
             [4.88, 180.36, 361.36, 547.8, 0],
             [4.95, 180.43, 361.44, 547.87, 0],
             [5.06, 180.54, 361.56, 547.98, 0],
             [5.21, 180.68, 361.72, 548.12, 0],
             [5.4, 180.86, 0, 0, 0]]
        )
        expected_speed = np.array(
            [[0, 0, 0, 0, 0],
             [0.15, 0.17, 0.19, 0.14, 0],
             [0.35, 0.37, 0.39, 0.34, 0],
             [0.54, 0.57, 0.59, 0.54, 0],
             [0.74, 0.7, 0.79, 0.71, 0],
             [0.94, 0.9, 0, 0, 0]]
        )

        np.testing.assert_array_almost_equal(pos, expected_pos)
        np.testing.assert_array_almost_equal(speed, expected_speed)

    def test_time_space_diagram_ring_road(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        emission_data = tsd.import_data_from_emission(
            os.path.join(dir_path, 'test_files/loop_230_emission.csv'))

        flow_params = tsd.get_flow_params(
            os.path.join(dir_path, 'test_files/loop_230.json'))
        pos, speed, _ = tsd.get_time_space_data(emission_data, flow_params)

        expected_pos = np.array(
            [[0.0000e+00, 9.5500e+00, 9.5450e+01, 1.0500e+02, 1.1455e+02,
              1.2409e+02, 1.3364e+02, 1.4318e+02, 1.5273e+02, 1.6227e+02,
              1.7182e+02, 1.8136e+02, 1.9090e+01, 1.9091e+02, 2.0045e+02,
              2.8640e+01, 3.8180e+01, 4.7730e+01, 5.7270e+01, 6.6820e+01,
              7.6360e+01, 8.5910e+01],
             [1.0000e-02, 9.5500e+00, 9.5460e+01, 1.0501e+02, 1.1455e+02,
              1.2410e+02, 1.3364e+02, 1.4319e+02, 1.5274e+02, 1.6228e+02,
              1.7183e+02, 1.8137e+02, 1.9100e+01, 1.9092e+02, 2.0046e+02,
              2.8640e+01, 3.8190e+01, 4.7740e+01, 5.7280e+01, 6.6830e+01,
              7.6370e+01, 8.5920e+01],
             [2.0000e-02, 9.5700e+00, 9.5480e+01, 1.0502e+02, 1.1457e+02,
              1.2411e+02, 1.3366e+02, 1.4321e+02, 1.5275e+02, 1.6230e+02,
              1.7184e+02, 1.8139e+02, 1.9110e+01, 1.9093e+02, 2.0048e+02,
              2.8660e+01, 3.8210e+01, 4.7750e+01, 5.7300e+01, 6.6840e+01,
              7.6390e+01, 8.5930e+01],
             [5.0000e-02, 9.5900e+00, 9.5500e+01, 1.0505e+02, 1.1459e+02,
              1.2414e+02, 1.3368e+02, 1.4323e+02, 1.5277e+02, 1.6232e+02,
              1.7187e+02, 1.8141e+02, 1.9140e+01, 1.9096e+02, 2.0051e+02,
              2.8680e+01, 3.8230e+01, 4.7770e+01, 5.7320e+01, 6.6870e+01,
              7.6410e+01, 8.5960e+01],
             [8.0000e-02, 9.6200e+00, 9.5530e+01, 1.0508e+02, 1.1462e+02,
              1.2417e+02, 1.3371e+02, 1.4326e+02, 1.5281e+02, 1.6235e+02,
              1.7190e+02, 1.8144e+02, 1.9170e+01, 1.9099e+02, 2.0055e+02,
              2.8710e+01, 3.8260e+01, 4.7810e+01, 5.7350e+01, 6.6900e+01,
              7.6440e+01, 8.5990e+01],
             [1.2000e-01, 9.6600e+00, 9.5570e+01, 1.0512e+02, 1.1466e+02,
              1.2421e+02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
              0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
              0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
              0.0000e+00, 0.0000e+00]]
        )
        expected_speed = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08,
             0.08, 0.08, 0.08, 0.1, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08],
            [0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16,
             0.16, 0.16, 0.16, 0.2, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16],
            [0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23,
             0.23, 0.23, 0.23, 0.29, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23],
            [0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31,
             0.31, 0.31, 0.31, 0.39, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31],
            [0.41, 0.41, 0.41, 0.41, 0.41, 0.41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0]
        ])

        np.testing.assert_array_almost_equal(pos, expected_pos)
        np.testing.assert_array_almost_equal(speed, expected_speed)

    def test_plot_ray_results(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(dir_path, 'test_files/progress.csv')

        parser = prr.create_parser()

        # test with one column
        args = parser.parse_args([file_path, 'episode_reward_mean'])
        prr.plot_progress(args.file, args.columns)

        # test with several columns
        args = parser.parse_args([file_path, 'episode_reward_mean',
                                  'episode_reward_min', 'episode_reward_max'])
        prr.plot_progress(args.file, args.columns)

        # test with non-existing column name
        with self.assertRaises(KeyError):
            args = parser.parse_args([file_path, 'episode_reward'])
            prr.plot_progress(args.file, args.columns)

        # test with column containing non-float values
        with self.assertRaises(ValueError):
            args = parser.parse_args([file_path, 'info'])
            prr.plot_progress(args.file, args.columns)

        # test that script outputs available column names if none is given
        column_names = [
            'episode_reward_max',
            'episode_reward_min',
            'episode_reward_mean',
            'episode_len_mean',
            'episodes_this_iter',
            'policy_reward_mean',
            'custom_metrics',
            'sampler_perf',
            'off_policy_estimator',
            'num_metric_batches_dropped',
            'info',
            'timesteps_this_iter',
            'done',
            'timesteps_total',
            'episodes_total',
            'training_iteration',
            'experiment_id',
            'date',
            'timestamp',
            'time_this_iter_s',
            'time_total_s',
            'pid',
            'hostname',
            'node_ip',
            'config',
            'time_since_restore',
            'timesteps_since_restore',
            'iterations_since_restore'
        ]

        temp_stdout = StringIO()
        with contextlib.redirect_stdout(temp_stdout):
            args = parser.parse_args([file_path])
            prr.plot_progress(args.file, args.columns)
        output = temp_stdout.getvalue()

        for column in column_names:
            self.assertTrue(column in output)


if __name__ == '__main__':
    ray.init(num_cpus=1)
    unittest.main()
    ray.shutdown()
