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
        dir_path = os.path.dirname(os.path.realpath(__file__))
        flow_params = tsd.get_flow_params(
            os.path.join(dir_path, 'test_files/fig8.json'))
        emission_data = tsd.import_data_from_trajectory(
            os.path.join(dir_path, 'test_files/fig8_emission.csv'), flow_params)

        segs, _ = tsd.get_time_space_data(emission_data, flow_params)

        expected_segs = np.array([
          [[1., 60.], [2., 59.]],
          [[2., 59.], [3., 57.02]],
          [[3., 57.02], [4., 54.05]],
          [[1., 23.8], [2., 22.81]],
          [[2., 22.81], [3., 20.83]],
          [[3., 20.83], [4., 17.89]],
          [[1., 182.84166941], [2., 181.85166941]],
          [[2., 181.85166941], [3., 179.87166941]],
          [[3., 179.87166941], [4., 176.92166941]],
          [[1., 154.07166941], [2., 153.08166941]],
          [[2., 153.08166941], [3., 151.10166941]],
          [[3., 151.10166941], [4., 148.16166941]],
          [[1., 125.30166941], [2., 124.31166941]],
          [[2., 124.31166941], [3., 122.34166941]],
          [[3., 122.34166941], [4., 119.39166941]],
          [[1., 96.54166941], [2., 95.54166941]],
          [[2., 95.54166941], [3., 93.56166941]],
          [[3., 93.56166941], [4., 90.59166941]],
          [[1., -203.16166941], [2., -202.17166941]],
          [[2., -202.17166941], [3., -200.02166941]],
          [[3., -200.02166941], [4., -197.07166941]],
          [[1., -174.40166941], [2., -173.40166941]],
          [[2., -173.40166941], [3., -171.43166941]],
          [[3., -171.43166941], [4., -168.48166941]],
          [[1., -145.63166941], [2., -144.64166941]],
          [[2., -144.64166941], [3., -142.66166941]],
          [[3., -142.66166941], [4., -139.72166941]],
          [[1., -116.86166941], [2., -115.87166941]],
          [[2., -115.87166941], [3., -113.89166941]],
          [[3., -113.89166941], [4., -110.95166941]],
          [[1., -88.09166941], [2., -87.10166941]],
          [[2., -87.10166941], [3., -85.13166941]],
          [[3., -85.13166941], [4., -82.18166941]],
          [[1., -59.33], [2., -58.34]],
          [[2., -58.34], [3., -56.36]],
          [[3., -56.36], [4., -53.42]],
          [[1., -30.56], [2., -29.72]],
          [[2., -29.72], [3., -27.97]],
          [[3., -27.97], [4., -25.22]],
          [[1., -1.79], [2., -0.8]],
          [[2., -0.8], [3., 208.64166941]],
          [[3., 208.64166941], [4., 205.69166941]]]
        )

        np.testing.assert_array_almost_equal(segs, expected_segs)

    def test_time_space_diagram_merge(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        flow_params = tsd.get_flow_params(
            os.path.join(dir_path, 'test_files/merge.json'))
        emission_data = tsd.import_data_from_trajectory(
            os.path.join(dir_path, 'test_files/merge_emission.csv'), flow_params)

        segs, _ = tsd.get_time_space_data(emission_data, flow_params)

        expected_segs = np.array([
          [[2.0000e-01, 7.2949e+02], [4.0000e-01, 7.2953e+02]],
          [[4.0000e-01, 7.2953e+02], [6.0000e-01, 7.2961e+02]],
          [[6.0000e-01, 7.2961e+02], [8.0000e-01, 7.2973e+02]],
          [[8.0000e-01, 7.2973e+02], [1.0000e+00, 7.2988e+02]]]
        )

        np.testing.assert_array_almost_equal(segs, expected_segs)

    def test_time_space_diagram_I210(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        module = __import__("examples.exp_configs.non_rl", fromlist=["i210_subnetwork"])
        flow_params = getattr(module, "i210_subnetwork").flow_params
        emission_data = tsd.import_data_from_trajectory(
            os.path.join(dir_path, 'test_files/i210_emission.csv'), flow_params)

        segs, _ = tsd.get_time_space_data(emission_data, flow_params)

        expected_segs = {
          1: np.array([
            [[0.8, 5.1], [1.6, 23.37]],
            [[1.6, 23.37], [2.4, 42.02]],
            [[2.4, 42.02], [3.2, 61.21]],
            [[3.2, 61.21], [4., 18.87]],
            [[4., 18.87], [4.8, 39.93]],
            [[2.4, 5.1], [3.2, 22.97]],
            [[3.2, 22.97], [4., 40.73]]]
          ),
          2: np.array([
            [[2.4, 5.1], [3.2, 23.98]],
            [[3.2, 23.98], [4., 43.18]]]
          ),
          3: np.array([
            [[0.8, 5.1], [1.6, 23.72]],
            [[1.6, 23.72], [2.4, 43.06]],
            [[2.4, 43.06], [3.2, 1.33]],
            [[3.2, 1.33], [4., 21.65]],
            [[4., 21.65], [4.8, 43.46]],
            [[2.4, 5.1], [3.2, 23.74]],
            [[3.2, 23.74], [4., 42.38]]]
          ),
          4: np.array([
            [[2.4, 5.1], [3.2, 23.6]],
            [[3.2, 23.6], [4., 42.46]]]
          )}

        for lane, expected_seg in expected_segs.items():
            np.testing.assert_array_almost_equal(segs[lane], expected_seg)

    def test_time_space_diagram_ring_road(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        flow_params = tsd.get_flow_params(
            os.path.join(dir_path, 'test_files/ring_230.json'))
        emission_data = tsd.import_data_from_trajectory(
            os.path.join(dir_path, 'test_files/ring_230_emission.csv'), flow_params)

        segs, _ = tsd.get_time_space_data(emission_data, flow_params)

        expected_segs = np.array([
          [[1.0000e-01, 0.0000e+00], [2.0000e-01, 1.0000e-02]],
          [[2.0000e-01, 1.0000e-02], [3.0000e-01, 2.0000e-02]],
          [[3.0000e-01, 2.0000e-02], [4.0000e-01, 5.0000e-02]],
          [[4.0000e-01, 5.0000e-02], [5.0000e-01, 8.0000e-02]],
          [[5.0000e-01, 8.0000e-02], [6.0000e-01, 1.2000e-01]],
          [[1.0000e-01, 9.5500e+00], [2.0000e-01, 9.5500e+00]],
          [[2.0000e-01, 9.5500e+00], [3.0000e-01, 9.5700e+00]],
          [[3.0000e-01, 9.5700e+00], [4.0000e-01, 9.5900e+00]],
          [[4.0000e-01, 9.5900e+00], [5.0000e-01, 9.6200e+00]],
          [[5.0000e-01, 9.6200e+00], [6.0000e-01, 9.6600e+00]],
          [[1.0000e-01, 9.5550e+01], [2.0000e-01, 9.5560e+01]],
          [[2.0000e-01, 9.5560e+01], [3.0000e-01, 9.5580e+01]],
          [[3.0000e-01, 9.5580e+01], [4.0000e-01, 9.5600e+01]],
          [[4.0000e-01, 9.5600e+01], [5.0000e-01, 9.5630e+01]],
          [[5.0000e-01, 9.5630e+01], [6.0000e-01, 9.5670e+01]],
          [[1.0000e-01, 1.0510e+02], [2.0000e-01, 1.0511e+02]],
          [[2.0000e-01, 1.0511e+02], [3.0000e-01, 1.0512e+02]],
          [[3.0000e-01, 1.0512e+02], [4.0000e-01, 1.0515e+02]],
          [[4.0000e-01, 1.0515e+02], [5.0000e-01, 1.0518e+02]],
          [[5.0000e-01, 1.0518e+02], [6.0000e-01, 1.0522e+02]],
          [[1.0000e-01, 1.1465e+02], [2.0000e-01, 1.1465e+02]],
          [[2.0000e-01, 1.1465e+02], [3.0000e-01, 1.1467e+02]],
          [[3.0000e-01, 1.1467e+02], [4.0000e-01, 1.1469e+02]],
          [[4.0000e-01, 1.1469e+02], [5.0000e-01, 1.1472e+02]],
          [[5.0000e-01, 1.1472e+02], [6.0000e-01, 1.1476e+02]],
          [[1.0000e-01, 1.2429e+02], [2.0000e-01, 1.2430e+02]],
          [[2.0000e-01, 1.2430e+02], [3.0000e-01, 1.2431e+02]],
          [[3.0000e-01, 1.2431e+02], [4.0000e-01, 1.2434e+02]],
          [[4.0000e-01, 1.2434e+02], [5.0000e-01, 1.2437e+02]],
          [[5.0000e-01, 1.2437e+02], [6.0000e-01, 1.2441e+02]],
          [[1.0000e-01, 1.3384e+02], [2.0000e-01, 1.3384e+02]],
          [[2.0000e-01, 1.3384e+02], [3.0000e-01, 1.3386e+02]],
          [[3.0000e-01, 1.3386e+02], [4.0000e-01, 1.3388e+02]],
          [[4.0000e-01, 1.3388e+02], [5.0000e-01, 1.3391e+02]],
          [[1.0000e-01, 1.4338e+02], [2.0000e-01, 1.4339e+02]],
          [[2.0000e-01, 1.4339e+02], [3.0000e-01, 1.4341e+02]],
          [[3.0000e-01, 1.4341e+02], [4.0000e-01, 1.4343e+02]],
          [[4.0000e-01, 1.4343e+02], [5.0000e-01, 1.4346e+02]],
          [[1.0000e-01, 1.5293e+02], [2.0000e-01, 1.5294e+02]],
          [[2.0000e-01, 1.5294e+02], [3.0000e-01, 1.5295e+02]],
          [[3.0000e-01, 1.5295e+02], [4.0000e-01, 1.5297e+02]],
          [[4.0000e-01, 1.5297e+02], [5.0000e-01, 1.5301e+02]],
          [[1.0000e-01, 1.6247e+02], [2.0000e-01, 1.6248e+02]],
          [[2.0000e-01, 1.6248e+02], [3.0000e-01, 1.6250e+02]],
          [[3.0000e-01, 1.6250e+02], [4.0000e-01, 1.6252e+02]],
          [[4.0000e-01, 1.6252e+02], [5.0000e-01, 1.6255e+02]],
          [[1.0000e-01, 1.7202e+02], [2.0000e-01, 1.7203e+02]],
          [[2.0000e-01, 1.7203e+02], [3.0000e-01, 1.7204e+02]],
          [[3.0000e-01, 1.7204e+02], [4.0000e-01, 1.7207e+02]],
          [[4.0000e-01, 1.7207e+02], [5.0000e-01, 1.7210e+02]],
          [[1.0000e-01, 1.8166e+02], [2.0000e-01, 1.8167e+02]],
          [[2.0000e-01, 1.8167e+02], [3.0000e-01, 1.8169e+02]],
          [[3.0000e-01, 1.8169e+02], [4.0000e-01, 1.8171e+02]],
          [[4.0000e-01, 1.8171e+02], [5.0000e-01, 1.8174e+02]],
          [[1.0000e-01, 1.9090e+01], [2.0000e-01, 1.9100e+01]],
          [[2.0000e-01, 1.9100e+01], [3.0000e-01, 1.9110e+01]],
          [[3.0000e-01, 1.9110e+01], [4.0000e-01, 1.9140e+01]],
          [[4.0000e-01, 1.9140e+01], [5.0000e-01, 1.9170e+01]],
          [[1.0000e-01, 1.9121e+02], [2.0000e-01, 1.9122e+02]],
          [[2.0000e-01, 1.9122e+02], [3.0000e-01, 1.9123e+02]],
          [[3.0000e-01, 1.9123e+02], [4.0000e-01, 1.9126e+02]],
          [[4.0000e-01, 1.9126e+02], [5.0000e-01, 1.9129e+02]],
          [[1.0000e-01, 2.0075e+02], [2.0000e-01, 2.0076e+02]],
          [[2.0000e-01, 2.0076e+02], [3.0000e-01, 2.0078e+02]],
          [[3.0000e-01, 2.0078e+02], [4.0000e-01, 2.0081e+02]],
          [[4.0000e-01, 2.0081e+02], [5.0000e-01, 2.0085e+02]],
          [[1.0000e-01, 2.8640e+01], [2.0000e-01, 2.8640e+01]],
          [[2.0000e-01, 2.8640e+01], [3.0000e-01, 2.8660e+01]],
          [[3.0000e-01, 2.8660e+01], [4.0000e-01, 2.8680e+01]],
          [[4.0000e-01, 2.8680e+01], [5.0000e-01, 2.8710e+01]],
          [[1.0000e-01, 3.8180e+01], [2.0000e-01, 3.8190e+01]],
          [[2.0000e-01, 3.8190e+01], [3.0000e-01, 3.8210e+01]],
          [[3.0000e-01, 3.8210e+01], [4.0000e-01, 3.8230e+01]],
          [[4.0000e-01, 3.8230e+01], [5.0000e-01, 3.8260e+01]],
          [[1.0000e-01, 4.7730e+01], [2.0000e-01, 4.7740e+01]],
          [[2.0000e-01, 4.7740e+01], [3.0000e-01, 4.7750e+01]],
          [[3.0000e-01, 4.7750e+01], [4.0000e-01, 4.7770e+01]],
          [[4.0000e-01, 4.7770e+01], [5.0000e-01, 4.7810e+01]],
          [[1.0000e-01, 5.7270e+01], [2.0000e-01, 5.7280e+01]],
          [[2.0000e-01, 5.7280e+01], [3.0000e-01, 5.7300e+01]],
          [[3.0000e-01, 5.7300e+01], [4.0000e-01, 5.7320e+01]],
          [[4.0000e-01, 5.7320e+01], [5.0000e-01, 5.7350e+01]],
          [[1.0000e-01, 6.6920e+01], [2.0000e-01, 6.6930e+01]],
          [[2.0000e-01, 6.6930e+01], [3.0000e-01, 6.6940e+01]],
          [[3.0000e-01, 6.6940e+01], [4.0000e-01, 6.6970e+01]],
          [[4.0000e-01, 6.6970e+01], [5.0000e-01, 6.7000e+01]],
          [[1.0000e-01, 7.6460e+01], [2.0000e-01, 7.6470e+01]],
          [[2.0000e-01, 7.6470e+01], [3.0000e-01, 7.6490e+01]],
          [[3.0000e-01, 7.6490e+01], [4.0000e-01, 7.6510e+01]],
          [[4.0000e-01, 7.6510e+01], [5.0000e-01, 7.6540e+01]],
          [[1.0000e-01, 8.6010e+01], [2.0000e-01, 8.6020e+01]],
          [[2.0000e-01, 8.6020e+01], [3.0000e-01, 8.6030e+01]],
          [[3.0000e-01, 8.6030e+01], [4.0000e-01, 8.6060e+01]],
          [[4.0000e-01, 8.6060e+01], [5.0000e-01, 8.6090e+01]]]
        )

        np.testing.assert_array_almost_equal(segs, expected_segs)

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
