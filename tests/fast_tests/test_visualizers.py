import unittest
import os
import pickle
import numpy as np
from flow.visualize.visualizer_rllab import visualizer_rllab
from flow.visualize.visualizer_rllib import visualizer_rllib

os.environ["TEST_FLAG"] = "True"


class TestVisualizerRLlib(unittest.TestCase):
    """
    Tests visualizer_flow:
    - ensures that it runs
    """

    # TODO fix this test
    def test_visualizer(self):
        # current path
        current_path = os.path.realpath(__file__).rsplit("/", 1)[0]

        # run the experiment and check it doesn't crash
        # FIXME(ev) it's not actually catching errors
        # convert os into a method

        os.system("python %s/../../flow/visualize/visualizer_rllib.py "
                  "%s/../data/rllib_data/ 1 --num_rollouts 1 "
                  "--no_render" %
                  (current_path, current_path))




class TestVisualizerRLlab(unittest.TestCase):
    """
    Tests visualizer_flow:
    - ensures that it runs
    """

    # TODO fix this test
    def test_visualizer(self):
        # current path
        current_path = os.path.realpath(__file__).rsplit("/", 1)[0]

        # run the experiment and check it doesn't crash
        # FIXME(ev) it's not actually catching errors
        try:
            os.system("python %s/../../flow/visualize/visualizer_rllab.py "
                      "%s/../data/rllab_data/itr_0.pkl --num_rollouts 1 "
                      "--no_render" %
                      (current_path, current_path))
        except Exception as e:
            self.assert_(False)


if __name__ == '__main__':
    unittest.main()
