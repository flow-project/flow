"""Unit tests for macroscopic models."""
import unittest
import os
from flow.core.macroscopic import LWR, ARZ
from flow.core.macroscopic.lwr import PARAMS as LWR_PARAMS
from flow.core.macroscopic.arz import PARAMS as ARZ_PARAMS
from flow.core.macroscopic.utils import DictDescriptor

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

#
# class TestLWR(unittest.TestCase):
#     """Unit tests for the flow.core.macroscopic.lwr.LWR class."""
#
#     def test_init(self):
#         # test the dt/total_time assertion
#         params = LWR_PARAMS.copy()
#         params['total_time'] = 10
#         params['dt'] = 3
#         self.assertRaises(AssertionError, LWR, params=params)

#         # test the dx/length assertion
#         params = LWR_PARAMS.copy()
#         params['length'] = 10
#         params['dx'] = 3
#         self.assertRaises(AssertionError, LWR, params=params)
#
#         # test the v_max/v_max_max assertion
#         params = LWR_PARAMS.copy()
#         params['v_max'] = 10
#         params['v_max_max'] = 3
#         self.assertRaises(AssertionError, LWR, params=params)
#
#         # test the rho_max/rho_max_max assertion
#         params = LWR_PARAMS.copy()
#         params['rho_max'] = 10
#         params['rho_max_max'] = 3
#         self.assertRaises(AssertionError, LWR, params=params)
#
#         # test the dt/dx/CFL/v_max assertion
#         pass
#
#         # check the action space
#         pass
#
#         # check the observation space
#         pass
#
#         # validate that all the inputs properly match the expected values
#         pass
#
#     def test_speed_info(self):
#         # test the implementation of the Greenshields model
#         pass
#
#     def test_IBVP(self):
#         # test the implementation of the Godunov scheme for multi-populations
#         pass
#
#     def test_step(self):
#         # check that the output from the step method matches expected values
#         pass
#
#     def test_reset(self):
#         env = LWR(LWR_PARAMS.copy())
#
#         # check that the initial v_max value matches the expected term and it
#         # had been initially changed to another value
#         env.v_max = 5
#         env.reset()
#         self.assertEqual(env.v_max, 27.5)
#
#         # check that the initial density and observations match the expected
#         # values
#         pass


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
