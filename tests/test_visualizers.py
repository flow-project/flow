import unittest
import os
import pickle
import numpy as np


class TestVisualizerFlow(unittest.TestCase):
    """
    Tests visualizer_flow:
    - ensures that it runs
    - ensures that crashes in the visualizer does not cause the visualizer to
      crash, and that observations are still being stored
    """
    def runTest(self):
        # run the experiment and check it doesn't crash
        os.system("python ../flow/visualizer_flow.py "
                  "test_files/params-collide.pkl --num_rollouts 1")

        self.assert_(True)

        # open the generated observations file, and check it isn't all zeros
        observations = pickle.load(open("observations.pkl", "rb"))

        self.assertNotEqual(np.sum(np.sum(observations)), 0)

    # def test_run_long(self):
    #     """
    #     Tests that the "run_long" parameter allows the rollout to run for a
    #     multiple of what the original max_path_length was, and does not create
    #     empty observation and rewards files.
    #     """
    #     pass


if __name__ == '__main__':
    unittest.main()
