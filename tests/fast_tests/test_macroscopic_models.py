"""Unit tests for macroscopic models."""
import unittest
import os
from flow.core.macroscopic import LWR, ARZ
from flow.core.macroscopic.lwr import PARAMS as LWR_PARAMS
from flow.core.macroscopic.arz import PARAMS as ARZ_PARAMS
from flow.core.macroscopic.utils import DictDescriptor
from flow.core.macroscopic.run_macro_model import parse_args
from flow.core.macroscopic.run_macro_model import parse_model_args
from flow.core.macroscopic.run_macro_model import load_model_env

os.environ["TEST_FLAG"] = "True"


class TestDictDescriptor(unittest.TestCase):
    """Unit tests for the flow.core.macroscopic.utils.DictDescriptor class."""

    def test_dict_descriptor(self):
        test_dict = DictDescriptor(
            ("foo", 1, float, "foofoo"),
            ("bar", True, bool, "barbar"),
        )

        # test the copy method
        self.assertDictEqual(test_dict.copy(), {"foo": 1, "bar": True})

        # test the description method
        self.assertEqual(test_dict.description("foo"), "foofoo")
        self.assertEqual(test_dict.description("bar"), "barbar")

        # test the type method
        self.assertEqual(test_dict.type("foo"), float)
        self.assertEqual(test_dict.type("bar"), bool)


class TestRunMacroModel(unittest.TestCase):
    """Unit tests for the file flow/core/macroscopic/run_macro_model.py."""

    def test_parse_args(self):
        # default case
        flags = parse_args(['LWR'])
        expected_results = {
            'model_name': 'LWR',
            'simulate': False,
            'plot_results': False,
            'save_results': False,
            'save_path': None,
            'checkpoint_path': None,
            'checkpoint_num': None,
            'include_params': False,
            'train': False,
            'n_rollouts': 1,
            'n_cpus': 1,
            'seed': 1,
            'n_itr': 1
        }
        self.assertDictEqual(vars(flags), expected_results)

        # with arguments
        flags = parse_args(['LWR', '--simulate'])
        self.assertTrue(flags.simulate)

        flags = parse_args(['LWR', '--train'])
        self.assertTrue(flags.train)

        flags = parse_args(['LWR', '--plot_results'])
        self.assertTrue(flags.plot_results)

        flags = parse_args(['LWR', '--save_results'])
        self.assertTrue(flags.save_results)

        flags = parse_args(['LWR', '--include_params'])
        self.assertTrue(flags.include_params)

        flags = parse_args([
            'LWR',
            '--save_path', 'foo',
            '--n_rollouts', '1',
            '--n_cpus', '2',
            '--seed', '3',
            '--n_itr', '4'
        ])
        self.assertEqual(flags.save_path, 'foo')
        self.assertEqual(flags.n_rollouts, 1)
        self.assertEqual(flags.n_cpus, 2)
        self.assertEqual(flags.seed, 3)
        self.assertEqual(flags.n_itr, 4)

    def test_parse_model_args(self):
        # test the LWR case
        flags = parse_model_args([], 'LWR')
        expected_results = {
            'length': 10000,
            'dx': 100,
            'rho_max': 0.2,
            'rho_max_max': 0.2,
            'v_max': 27.5,
            'v_max_max': 27.5,
            'CFL': 0.95,
            'total_time': 500,
            'dt': 1
        }
        self.assertDictEqual(vars(flags), expected_results)

        # test the ARZ case
        flags = parse_model_args([], 'ARZ')
        expected_results = {
            'length': 10000,
            'dx': 100,
            'rho_max': 0.2,
            'rho_max_max': 0.2,
            'v_max': 27.5,
            'v_max_max': 27.5,
            'CFL': 0.95,
            'tau': 0.1,
            'total_time': 500,
            'dt': 1
        }
        self.assertDictEqual(vars(flags), expected_results)

    def test_load_model_env(self):
        # test the default case for LWR
        env, agent = load_model_env('LWR')
        self.assertIsNone(agent)
        self.assertDictEqual(env.params, LWR_PARAMS.copy())

        # test the default case for ARZ
        env, agent = load_model_env('ARZ')
        self.assertIsNone(agent)
        self.assertDictEqual(env.params, ARZ_PARAMS.copy())

        # test the updating model parameters
        env, agent = load_model_env('ARZ', model_params={'tau': 100})
        self.assertIsNone(agent)
        self.assertEqual(env.tau, 100)

        # test importing model from checkpoint
        pass

        # test importing model parameters from checkpoint
        pass

    def test_rollout(self):
        pass

    def test_run_training(self):
        pass


class TestLWR(unittest.TestCase):
    """Unit tests for the flow.core.macroscopic.lwr.LWR class."""

    def test_init(self):
        # test the dt/total_time assertion
        params = LWR_PARAMS.copy()
        params['total_time'] = 10
        params['dt'] = 3
        self.assertRaises(AssertionError, LWR, params=params)

        # test the dx/length assertion
        params = LWR_PARAMS.copy()
        params['length'] = 10
        params['dx'] = 3
        self.assertRaises(AssertionError, LWR, params=params)

        # test the v_max/v_max_max assertion
        params = LWR_PARAMS.copy()
        params['v_max'] = 10
        params['v_max_max'] = 3
        self.assertRaises(AssertionError, LWR, params=params)

        # test the rho_max/rho_max_max assertion
        params = LWR_PARAMS.copy()
        params['rho_max'] = 10
        params['rho_max_max'] = 3
        self.assertRaises(AssertionError, LWR, params=params)

        # test the dt/dx/CFL/v_max assertion
        pass

        # check the action space
        pass

        # check the observation space
        pass

        # validate that all the inputs properly match the expected values
        pass

    def test_speed_info(self):
        # test the implementation of the Greenshields model
        pass

    def test_IBVP(self):
        # test the implementation of the Godunov scheme for multi-populations
        pass

    def test_step(self):
        # check that the output from the step method matches expected values
        pass

    def test_reset(self):
        env = LWR(LWR_PARAMS.copy())

        # check that the initial v_max value matches the expected term and it
        # had been initially changed to another value
        env.v_max = 5
        env.reset()
        self.assertEqual(env.v_max, 27.5)

        # check that the initial density and observations match the expected
        # values
        pass


class TestARZ(unittest.TestCase):
    """Unit tests for the flow.core.macroscopic.arz.ARZ class."""

    def test_init(self):
        # test the dt/total_time assertion
        params = ARZ_PARAMS.copy()
        params['total_time'] = 10
        params['dt'] = 3
        self.assertRaises(AssertionError, ARZ, params=params)

        # test the dx/length assertion
        params = ARZ_PARAMS.copy()
        params['length'] = 10
        params['dx'] = 3
        self.assertRaises(AssertionError, ARZ, params=params)

        # test the v_max/v_max_max assertion
        params = ARZ_PARAMS.copy()
        params['v_max'] = 10
        params['v_max_max'] = 3
        self.assertRaises(AssertionError, ARZ, params=params)

        # test the rho_max/rho_max_max assertion
        params = ARZ_PARAMS.copy()
        params['rho_max'] = 10
        params['rho_max_max'] = 3
        self.assertRaises(AssertionError, ARZ, params=params)

        # test the dt/dx/CFL/v_max assertion
        pass

        # check the action space
        pass

        # check the observation space
        pass

        # validate that all the inputs properly match the expected values
        pass

    def test_step(self):
        # check that the output from the step method matches expected values
        pass

    def test_reset(self):
        env = ARZ(ARZ_PARAMS.copy())

        # check that the initial v_max value matches the expected term and it
        # had been initially changed to another value
        env.v_max = 5
        env.reset()
        self.assertEqual(env.v_max, 27.5)

        # check that the initial density and observations match the expected
        # values
        pass


if __name__ == '__main__':
    unittest.main()
