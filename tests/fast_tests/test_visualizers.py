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
        emission_data, _, _, _ = tsd.import_data_from_trajectory(
            os.path.join(dir_path, 'test_files/fig8_emission.csv'), flow_params)

        segs, _ = tsd.get_time_space_data(emission_data, flow_params['network'])

        expected_segs = np.array([
          [[1., 263.16166941], [2., 262.16166941]],
          [[2., 262.16166941], [3., 260.18166941]],
          [[3., 260.18166941], [4., 257.21166941]],
          [[1., 226.96166941], [2., 225.97166941]],
          [[2., 225.97166941], [3., 223.99166941]],
          [[3., 223.99166941], [4., 221.05166941]],
          [[1., 386.00333882], [2., 385.01333882]],
          [[2., 385.01333882], [3., 383.03333882]],
          [[3., 383.03333882], [4., 380.08333882]],
          [[1., 357.23333882], [2., 356.24333882]],
          [[2., 356.24333882], [3., 354.26333882]],
          [[3., 354.26333882], [4., 351.32333882]],
          [[1., 328.46333882], [2., 327.47333882]],
          [[2., 327.47333882], [3., 325.50333882]],
          [[3., 325.50333882], [4., 322.55333882]],
          [[1., 299.70333882], [2., 298.70333882]],
          [[2., 298.70333882], [3., 296.72333882]],
          [[3., 296.72333882], [4., 293.75333882]],
          [[1., 0.], [2., 0.99]],
          [[2., 0.99], [3., 3.14]],
          [[3., 3.14], [4., 6.09]],
          [[1., 28.76], [2., 29.76]],
          [[2., 29.76], [3., 31.73]],
          [[3., 31.73], [4., 34.68]],
          [[1., 57.53], [2., 58.52]],
          [[2., 58.52], [3., 60.5]],
          [[3., 60.5], [4., 63.44]],
          [[1., 86.3], [2., 87.29]],
          [[2., 87.29], [3., 89.27]],
          [[3., 89.27], [4., 92.21]],
          [[1., 115.07], [2., 116.06]],
          [[2., 116.06], [3., 118.03]],
          [[3., 118.03], [4., 120.98]],
          [[1., 143.83166941], [2., 144.82166941]],
          [[2., 144.82166941], [3., 146.80166941]],
          [[3., 146.80166941], [4., 149.74166941]],
          [[1., 172.60166941], [2., 173.44166941]],
          [[2., 173.44166941], [3., 175.19166941]],
          [[3., 175.19166941], [4., 177.94166941]],
          [[1., 201.37166941], [2., 202.36166941]],
          [[2., 202.36166941], [3., 411.80333882]],
          [[3., 411.80333882], [4., 408.85333882]]]
        )

        np.testing.assert_array_almost_equal(segs, expected_segs)

    def test_time_space_diagram_merge(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        flow_params = tsd.get_flow_params(
            os.path.join(dir_path, 'test_files/merge.json'))
        emission_data, _, _, _ = tsd.import_data_from_trajectory(
            os.path.join(dir_path, 'test_files/merge_emission.csv'), flow_params)

        segs, _ = tsd.get_time_space_data(emission_data, flow_params['network'])

        expected_segs = np.array([
          [[2.0000e-01, 7.2463e+02], [4.0000e-01, 7.2467e+02]],
          [[4.0000e-01, 7.2467e+02], [6.0000e-01, 7.2475e+02]],
          [[6.0000e-01, 7.2475e+02], [8.0000e-01, 7.2487e+02]],
          [[8.0000e-01, 7.2487e+02], [1.0000e+00, 7.2502e+02]]]
        )

        np.testing.assert_array_almost_equal(segs, expected_segs)

    def test_time_space_diagram_I210(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        module = __import__("examples.exp_configs.non_rl", fromlist=["i210_subnetwork"])
        flow_params = getattr(module, "i210_subnetwork").flow_params
        emission_data, _, _, _ = tsd.import_data_from_trajectory(
            os.path.join(dir_path, 'test_files/i210_emission.csv'), flow_params)

        segs, _ = tsd.get_time_space_data(emission_data, flow_params['network'])

        expected_segs = {
          1: np.array([
            [[-719.2, 3.77], [-718.4,  22.04]],
            [[-718.4, 22.04], [-717.6, 40.69]],
            [[-717.6, 40.69], [-716.8, 59.88]],
            [[-716.8, 59.88], [-716., 17.54]],
            [[-716., 17.54], [-715.2, 38.6]],
            [[-717.6, 3.77], [-716.8, 21.64]],
            [[-716.8, 21.64], [-716., 39.4]]]
          ),
          2: np.array([
            [[-717.6, 3.77], [-716.8, 22.65]],
            [[-716.8, 22.65], [-716., 41.85]]]
          ),
          3: np.array([
            [[-719.2, 3.77], [-718.4, 22.39]],
            [[-718.4, 22.39], [-717.6, 41.73]],
            [[-717.6, 41.73], [-716.8, 0.]],
            [[-716.8, 0.], [-716., 20.32]],
            [[-716., 20.32], [-715.2, 42.13]],
            [[-717.6, 3.77], [-716.8, 22.41]],
            [[-716.8, 22.41], [-716., 41.05]]]
          ),
          4: np.array([
            [[-717.6, 3.77], [-716.8, 22.27]],
            [[-716.8, 22.27], [-716., 41.13]]]
          )}

        for lane, expected_seg in expected_segs.items():
            np.testing.assert_array_almost_equal(segs[lane], expected_seg)

    def test_time_space_diagram_ring_road(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        flow_params = tsd.get_flow_params(
            os.path.join(dir_path, 'test_files/ring_230.json'))
        emission_data, _, _, _ = tsd.import_data_from_trajectory(
            os.path.join(dir_path, 'test_files/ring_230_emission.csv'), flow_params)

        segs, _ = tsd.get_time_space_data(emission_data, flow_params['network'])

        expected_segs = np.array([
          [[-7.50000000e+01, 0.00000000e+00], [-7.49000000e+01, 7.98415842e-03]],
          [[-7.49000000e+01, 7.98415842e-03], [-7.48000000e+01, 2.37963776e-02]],
          [[-7.48000000e+01, 2.37963776e-02], [-7.47000000e+01, 4.72776801e-02]],
          [[-7.50000000e+01, 9.54545455e+00], [-7.49000000e+01, 9.55343870e+00]],
          [[-7.49000000e+01, 9.55343870e+00], [-7.48000000e+01, 9.56925092e+00]],
          [[-7.48000000e+01, 9.56925092e+00], [-7.47000000e+01, 9.59273223e+00]],
          [[-7.50000000e+01, 1.90909091e+01], [-7.49000000e+01, 1.90988932e+01]],
          [[-7.49000000e+01, 1.90988932e+01], [-7.48000000e+01, 1.91147055e+01]],
          [[-7.48000000e+01, 1.91147055e+01], [-7.47000000e+01, 1.91381868e+01]],
          [[-7.50000000e+01, 2.86363636e+01], [-7.49000000e+01, 2.86443478e+01]],
          [[-7.49000000e+01, 2.86443478e+01], [-7.48000000e+01, 2.86601600e+01]],
          [[-7.48000000e+01, 2.86601600e+01], [-7.47000000e+01, 2.86836413e+01]],
          [[-7.50000000e+01, 3.81818182e+01], [-7.49000000e+01, 3.81898023e+01]],
          [[-7.49000000e+01, 3.81898023e+01], [-7.48000000e+01, 3.82056146e+01]],
          [[-7.48000000e+01, 3.82056146e+01], [-7.47000000e+01, 3.82290959e+01]],
          [[-7.50000000e+01, 4.77272727e+01], [-7.49000000e+01, 4.77352569e+01]],
          [[-7.49000000e+01, 4.77352569e+01], [-7.48000000e+01, 4.77510691e+01]],
          [[-7.48000000e+01, 4.77510691e+01], [-7.47000000e+01, 4.77745504e+01]]]
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
