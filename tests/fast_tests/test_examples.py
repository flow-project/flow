import os
import unittest

from examples.sumo.bay_bridge import bay_bridge_example
from examples.sumo.bay_bridge_toll import bay_bridge_toll_example
from examples.sumo.bottleneck import bottleneck_example
from examples.sumo.figure_eight import figure_eight_example
from examples.sumo.grid import grid_example
from examples.sumo.highway import highway_example
from examples.sumo.loop_merge import loop_merge_example
from examples.sumo.merge import merge_example
from examples.sumo.sugiyama import sugiyama_example


os.environ['TEST_FLAG'] = 'True'


class TestSumoExamples(unittest.TestCase):
    """Tests the example scripts in examples/sumo.

    This is done by running the experiment function within each script for a
    few time steps. Note that, this does not test for any refactoring changes
    done to the functions within the experiment class.
    """

    def test_bottleneck(self):
        """Verifies that examples/sumo/bottleneck.py is working."""
        # import the experiment variable from the example
        exp = bottleneck_example(1000, 5, render=False)

        # run the experiment for a few time steps to ensure it doesn't fail
        exp.run(1, 5)

    def test_figure_eight(self):
        """Verifies that examples/sumo/figure_eight.py is working."""
        # import the experiment variable from the example
        exp = figure_eight_example(render=False)

        # run the experiment for a few time steps to ensure it doesn't fail
        exp.run(1, 5)

    def test_grid(self):
        """Verifies that examples/sumo/grid.py is working."""
        # import the experiment variable from the example
        exp = grid_example(render=False)

        # run the experiment for a few time steps to ensure it doesn't fail
        exp.run(1, 5)

    def test_highway(self):
        """Verifies that examples/sumo/highway.py is working."""
        # import the experiment variable from the example
        exp = highway_example(render=False)

        # run the experiment for a few time steps to ensure it doesn't fail
        exp.run(1, 5)

    def test_merge(self):
        """Verifies that examples/sumo/merge.py is working."""
        # import the experiment variable from the example
        exp = merge_example(render=False)

        # run the experiment for a few time steps to ensure it doesn't fail
        exp.run(1, 5)

    def test_sugiyama(self):
        """Verifies that examples/sumo/sugiyama.py is working."""
        # import the experiment variable from the example
        exp = sugiyama_example(render=False)

        # run the experiment for a few time steps to ensure it doesn't fail
        exp.run(1, 5)

    def test_loop_merge(self):
        """Verify that examples/sumo/two_loops_merge_straight.py is working."""
        # import the experiment variable from the example
        exp = loop_merge_example(render=False)

        # run the experiment for a few time steps to ensure it doesn't fail
        exp.run(1, 5)

    def test_bay_bridge(self):
        """Verifies that examples/sumo/bay_bridge.py is working."""
        # import the experiment variable from the example
        exp = bay_bridge_example(render=False)

        # run the experiment for a few time steps to ensure it doesn't fail
        exp.run(1, 5)

    def test_bay_bridge_toll(self):
        """Verifies that examples/sumo/bay_bridge_toll.py is working."""
        # import the experiment variable from the example
        exp = bay_bridge_toll_example(render=False)

        # run the experiment for a few time steps to ensure it doesn't fail
        exp.run(1, 5)


if __name__ == '__main__':
    unittest.main()
