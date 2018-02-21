import unittest
import os

from examples.sumo.cooperative_merge import cooperative_merge_example
from examples.sumo.figure_eight import figure_eight_example
from examples.sumo.highway import highway_example
from examples.sumo.loop_merge import loop_merge_example
from examples.sumo.sugiyama import sugiyama_example
from examples.sumo.two_lane_change_changer import two_lane_example
from examples.sumo.two_loops_merge import two_loops_merge_example
from examples.sumo.two_loops_merge_straight import two_loops_merge_straight_example
from examples.sumo.two_way_intersection import two_way_intersection_example


class TestSumoExamples(unittest.TestCase):
    """
    Tests the example scripts in examples/sumo. This is done by running the
    experiment function within each script for a few time steps. Note that, this
    does not test for any refactoring changes done to the functions within the
    experiment class.
    """

    def test_cooperative_merge(self):
        """
        Verifies that examples/sumo/cooperative_merge.py is working
        """
        # import the experiment variable from the example
        exp = cooperative_merge_example(sumo_binary="sumo")

        # run the experiment for a few time steps to ensure it doesn't fail
        exp.run(1, 5)

    def test_figure_eight(self):
        """
        Verifies that examples/sumo/figure_eight.py is working
        """
        # import the experiment variable from the example
        exp = figure_eight_example(sumo_binary="sumo")

        # run the experiment for a few time steps to ensure it doesn't fail
        exp.run(1, 5)

    def test_highway(self):
        """
        Verifies that examples/sumo/highway.py is working
        """
        # import the experiment variable from the example
        exp = highway_example(sumo_binary="sumo")

        # run the experiment for a few time steps to ensure it doesn't fail
        exp.run(1, 5)

    def test_loop_merge(self):
        """
        Verifies that examples/sumo/loop_merge.py is working
        """
        # import the experiment variable from the example
        exp = loop_merge_example(sumo_binary="sumo")

        # run the experiment for a few time steps to ensure it doesn't fail
        exp.run(1, 5)

    def test_sugiyama(self):
        """
        Verifies that examples/sumo/sugiyama.py is working
        """
        # import the experiment variable from the example
        exp = sugiyama_example(sumo_binary="sumo")

        # run the experiment for a few time steps to ensure it doesn't fail
        exp.run(1, 5)

    def test_two_lane_change_changer(self):
        """
        Verifies that examples/sumo/two_lane_change_changer.py is working
        """
        # import the experiment variable from the example
        exp = two_lane_example(sumo_binary="sumo")

        # run the experiment for a few time steps to ensure it doesn't fail
        exp.run(1, 5)

    def test_two_loops_merge(self):
        """
        Verifies that examples/sumo/two_loops_merge.py is working
        """
        # import the experiment variable from the example
        exp = two_loops_merge_example(sumo_binary="sumo")

        # run the experiment for a few time steps to ensure it doesn't fail
        exp.run(1, 5)

    def test_two_loops_merge_straight(self):
        """
        Verifies that examples/sumo/two_loops_merge_straight.py is working
        """
        # import the experiment variable from the example
        exp = two_loops_merge_straight_example(sumo_binary="sumo")

        # run the experiment for a few time steps to ensure it doesn't fail
        exp.run(1, 5)

    def test_two_way_intersection(self):
        """
        Verifies that examples/sumo/two_way_intersection.py is working
        """
        # import the experiment variable from the example
        exp = two_way_intersection_example(sumo_binary="sumo")

        # run the experiment for a few time steps to ensure it doesn't fail
        exp.run(1, 5)


if __name__ == '__main__':
    os.environ["TEST_FLAG"] = "True"
    unittest.main()
