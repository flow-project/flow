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
                               'upper_ring'],
                      'lane': [0.0, 0.0, 0.0, 0.0]},
            'idm_4': {'pos': [56.02, 57.01, 58.99, 61.93],
                      'time': [1.0, 2.0, 3.0, 4.0],
                      'vel': [0.0, 0.99, 1.98, 2.95],
                      'edge': ['upper_ring', 'upper_ring', 'upper_ring',
                               'upper_ring'],
                      'lane': [0.0, 0.0, 0.0, 0.0]},
            'idm_5': {'pos': [84.79, 85.78, 87.76, 90.7],
                      'time': [1.0, 2.0, 3.0, 4.0],
                      'vel': [0.0, 0.99, 1.98, 2.95],
                      'edge': ['upper_ring', 'upper_ring', 'upper_ring',
                               'upper_ring'],
                      'lane': [0.0, 0.0, 0.0, 0.0]},
            'idm_2': {'pos': [28.77, 29.76, 1.63, 4.58],
                      'time': [1.0, 2.0, 3.0, 4.0],
                      'vel': [0.0, 0.99, 1.97, 2.95],
                      'edge': ['top', 'top', 'upper_ring', 'upper_ring'],
                      'lane': [0.0, 0.0, 0.0, 0.0]},
            'idm_13': {'pos': [106.79, 107.79, 109.77, 112.74],
                       'time': [1.0, 2.0, 3.0, 4.0],
                       'vel': [0.0, 0.99, 1.98, 2.96],
                       'edge': ['lower_ring', 'lower_ring', 'lower_ring',
                                'lower_ring'],
                       'lane': [0.0, 0.0, 0.0, 0.0]},
            'idm_9': {'pos': [22.01, 23.0, 24.97, 27.92],
                      'time': [1.0, 2.0, 3.0, 4.0],
                      'vel': [0.0, 0.99, 1.97, 2.95],
                      'edge': ['left', 'left', 'left', 'left'],
                      'lane': [0.0, 0.0, 0.0, 0.0]},
            'idm_6': {'pos': [113.56, 114.55, 116.52, 119.47],
                      'time': [1.0, 2.0, 3.0, 4.0],
                      'vel': [0.0, 0.99, 1.97, 2.95],
                      'edge': ['upper_ring', 'upper_ring', 'upper_ring',
                               'upper_ring'],
                      'lane': [0.0, 0.0, 0.0, 0.0]},
            'idm_8': {'pos': [29.44, 0.28, 2.03, 4.78],
                      'time': [1.0, 2.0, 3.0, 4.0],
                      'vel': [0.0, 0.84, 1.76, 2.75],
                      'edge': ['right', ':center_0', ':center_0',
                               ':center_0'],
                      'lane': [0.0, 0.0, 0.0, 0.0]},
            'idm_12': {'pos': [78.03, 79.02, 80.99, 83.94],
                       'time': [1.0, 2.0, 3.0, 4.0],
                       'vel': [0.0, 0.99, 1.98, 2.95],
                       'edge': ['lower_ring', 'lower_ring', 'lower_ring',
                                'lower_ring'],
                       'lane': [0.0, 0.0, 0.0, 0.0]},
            'idm_10': {'pos': [20.49, 21.48, 23.46, 26.41],
                       'time': [1.0, 2.0, 3.0, 4.0],
                       'vel': [0.0, 0.99, 1.98, 2.95],
                       'edge': ['lower_ring', 'lower_ring', 'lower_ring',
                                'lower_ring'],
                       'lane': [0.0, 0.0, 0.0, 0.0]},
            'idm_11': {'pos': [49.26, 50.25, 52.23, 55.17],
                       'time': [1.0, 2.0, 3.0, 4.0],
                       'vel': [0.0, 0.99, 1.98, 2.95],
                       'edge': ['lower_ring', 'lower_ring', 'lower_ring',
                                'lower_ring'],
                       'lane': [0.0, 0.0, 0.0, 0.0]},
            'idm_1': {'pos': [0.0, 0.99, 2.97, 5.91],
                      'time': [1.0, 2.0, 3.0, 4.0],
                      'vel': [0.0, 0.99, 1.98, 2.95],
                      'edge': ['top', 'top', 'top', 'top'],
                      'lane': [0.0, 0.0, 0.0, 0.0]},
            'idm_7': {'pos': [0.67, 1.66, 3.64, 6.58],
                      'time': [1.0, 2.0, 3.0, 4.0],
                      'vel': [0.0, 0.99, 1.97, 2.94],
                      'edge': ['right', 'right', 'right', 'right'],
                      'lane': [0.0, 0.0, 0.0, 0.0]},
            'idm_0': {'pos': [0.0, 1.0, 2.98, 5.95],
                      'time': [1.0, 2.0, 3.0, 4.0],
                      'vel': [0.0, 1.0, 1.99, 2.97],
                      'edge': ['bottom', 'bottom', 'bottom', 'bottom'],
                      'lane': [0.0, 0.0, 0.0, 0.0]}
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
            os.path.join(dir_path, 'test_files/ring_230_emission.csv'))

        flow_params = tsd.get_flow_params(
            os.path.join(dir_path, 'test_files/ring_230.json'))
        pos, speed, _ = tsd.get_time_space_data(emission_data, flow_params)

        expected_pos = np.array(
            [[0.0000e+00, 9.5500e+00, 9.5550e+01, 1.0510e+02, 1.1465e+02,
              1.2429e+02, 1.3384e+02, 1.4338e+02, 1.5293e+02, 1.6247e+02,
              1.7202e+02, 1.8166e+02, 1.9090e+01, 1.9121e+02, 2.0075e+02,
              2.8640e+01, 3.8180e+01, 4.7730e+01, 5.7270e+01, 6.6920e+01,
              7.6460e+01, 8.6010e+01],
             [1.0000e-02, 9.5500e+00, 9.5560e+01, 1.0511e+02, 1.1465e+02,
              1.2430e+02, 1.3384e+02, 1.4339e+02, 1.5294e+02, 1.6248e+02,
              1.7203e+02, 1.8167e+02, 1.9100e+01, 1.9122e+02, 2.0076e+02,
              2.8640e+01, 3.8190e+01, 4.7740e+01, 5.7280e+01, 6.6930e+01,
              7.6470e+01, 8.6020e+01],
             [2.0000e-02, 9.5700e+00, 9.5580e+01, 1.0512e+02, 1.1467e+02,
              1.2431e+02, 1.3386e+02, 1.4341e+02, 1.5295e+02, 1.6250e+02,
              1.7204e+02, 1.8169e+02, 1.9110e+01, 1.9123e+02, 2.0078e+02,
              2.8660e+01, 3.8210e+01, 4.7750e+01, 5.7300e+01, 6.6940e+01,
              7.6490e+01, 8.6030e+01],
             [5.0000e-02, 9.5900e+00, 9.5600e+01, 1.0515e+02, 1.1469e+02,
              1.2434e+02, 1.3388e+02, 1.4343e+02, 1.5297e+02, 1.6252e+02,
              1.7207e+02, 1.8171e+02, 1.9140e+01, 1.9126e+02, 2.0081e+02,
              2.8680e+01, 3.8230e+01, 4.7770e+01, 5.7320e+01, 6.6970e+01,
              7.6510e+01, 8.6060e+01],
             [8.0000e-02, 9.6200e+00, 9.5630e+01, 1.0518e+02, 1.1472e+02,
              1.2437e+02, 1.3391e+02, 1.4346e+02, 1.5301e+02, 1.6255e+02,
              1.7210e+02, 1.8174e+02, 1.9170e+01, 1.9129e+02, 2.0085e+02,
              2.8710e+01, 3.8260e+01, 4.7810e+01, 5.7350e+01, 6.7000e+01,
              7.6540e+01, 8.6090e+01],
             [1.2000e-01, 9.6600e+00, 9.5670e+01, 1.0522e+02, 1.1476e+02,
              1.2441e+02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
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
