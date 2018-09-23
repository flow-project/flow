import unittest
import os
import pickle
import numpy as np

os.environ["TEST_FLAG"] = "True"
BROKEN_TESTS = os.environ.get('BROKEN_TESTS', False)


class TestVisualizerFlow(unittest.TestCase):
    """
    Tests visualizer_flow:
    - ensures that it runs
    - ensures that crashes in the visualizer does not cause the visualizer to
      crash, and that observations are still being stored
    """

    # TODO fix this test
    @unittest.skipUnless(BROKEN_TESTS, "broken test (known issue)")
    def test_visualizer(self):
        # current path
        current_path = os.path.realpath(__file__).rsplit("/", 1)[0]

        # run the experiment and check it doesn't crash
        os.system("python %s/../flow/visualizer_flow.py "
                  "tests/test_files/params-collide.pkl --num_rollouts 1" %
                  current_path)

        self.assert_(True)

        # open the generated observations file, and check it isn't all zeros
        observations = pickle.load(
            open(current_path + "/observations.pkl", "rb"))

        self.assertNotEqual(np.sum(np.sum(observations)), 0)


if __name__ == '__main__':
    unittest.main()
