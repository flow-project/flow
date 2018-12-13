import unittest
import os

from flow.benchmarks.baselines.bottleneck0 import bottleneck0_baseline
from flow.benchmarks.baselines.bottleneck1 import bottleneck1_baseline
from flow.benchmarks.baselines.bottleneck2 import bottleneck2_baseline
from flow.benchmarks.baselines.figureeight012 import figure_eight_baseline
from flow.benchmarks.baselines.grid0 import grid0_baseline
from flow.benchmarks.baselines.grid1 import grid1_baseline
from flow.benchmarks.baselines.merge012 import merge_baseline

os.environ["TEST_FLAG"] = "True"


class TestBaselines(unittest.TestCase):
    """
    Tests that the baselines in the benchmarks folder are running and
    returning expected values (i.e. values that match those in the CoRL paper
    reported on the website, or other).
    """

    def test_bottleneck0(self):
        """
        Tests flow/benchmark/baselines/bottleneck0.py
        """
        # run the bottleneck to make sure it runs
        bottleneck0_baseline(num_runs=1, render=False)

        # TODO: check that the performance measure is within some range

    def test_bottleneck1(self):
        """
        Tests flow/benchmark/baselines/bottleneck1.py
        """
        # run the bottleneck to make sure it runs
        bottleneck1_baseline(num_runs=1, render=False)

        # TODO: check that the performance measure is within some range

    def test_bottleneck2(self):
        """
        Tests flow/benchmark/baselines/bottleneck2.py
        """
        # run the bottleneck to make sure it runs
        bottleneck2_baseline(num_runs=1, render=False)

        # TODO: check that the performance measure is within some range

    def test_figure_eight(self):
        """
        Tests flow/benchmark/baselines/figureeight{0,1,2}.py
        """
        # run the bottleneck to make sure it runs
        figure_eight_baseline(num_runs=1, render=False)

        # TODO: check that the performance measure is within some range

    def test_grid0(self):
        """
        Tests flow/benchmark/baselines/grid0.py
        """
        # run the bottleneck to make sure it runs
        grid0_baseline(num_runs=1, render=False)

        # TODO: check that the performance measure is within some range

    def test_grid1(self):
        """
        Tests flow/benchmark/baselines/grid1.py
        """
        # run the bottleneck to make sure it runs
        grid1_baseline(num_runs=1, render=False)

        # TODO: check that the performance measure is within some range

    def test_merge(self):
        """
        Tests flow/benchmark/baselines/merge{0,1,2}.py
        """
        # run the bottleneck to make sure it runs
        merge_baseline(num_runs=1, render=False)

        # TODO: check that the performance measure is within some range


if __name__ == '__main__':
    unittest.main()
