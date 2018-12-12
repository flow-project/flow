import os
import unittest

# from examples.sumo.bay_bridge import bay_bridge_example
from examples.sumo.bay_bridge_toll import bay_bridge_toll_example
from examples.sumo.bottlenecks import bottleneck_example
from examples.sumo.figure_eight import figure_eight_example
from examples.sumo.grid import grid_example
from examples.sumo.highway import highway_example
from examples.sumo.loop_merge import loop_merge_example
from examples.sumo.merge import merge_example
from examples.sumo.minicity import minicity_example
from examples.sumo.sugiyama import sugiyama_example

from examples.rllib.cooperative_merge import setup_exps as coop_setup
from examples.rllib.figure_eight import setup_exps as figure_eight_setup
from examples.rllib.green_wave import setup_exps as green_wave_setup
from examples.rllib.stabilizing_highway import setup_exps as highway_setup
from examples.rllib.stabilizing_the_ring import setup_exps as ring_setup
from examples.rllib.velocity_bottleneck import setup_exps as bottleneck_setup
from examples.rllib.multiagent_exps.multiagent_figure_eight \
    import setup_exps as multi_figure_eight_setup
from examples.rllib.multiagent_exps.multiagent_stabilizing_the_ring \
    import setup_exps as multi_ring_setup

import ray
from ray.tune import run_experiments

os.environ['TEST_FLAG'] = 'True'


class TestSumoExamples(unittest.TestCase):
    """Tests the example scripts in examples/sumo.

    This is done by running the experiment function within each script for a
    few time steps. Note that, this does not test for any refactoring changes
    done to the functions within the experiment class.
    """

    def test_bottleneck(self):
        """Verifies that examples/sumo/bottlenecks.py is working."""
        # import the experiment variable from the example
        exp = bottleneck_example(20, 5, render=False)

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

    # def test_bay_bridge(self):
    #     """Verifies that examples/sumo/bay_bridge.py is working."""
    #     # import the experiment variable from the example
    #     exp = bay_bridge_example(render=False)
    #
    #     # run the experiment for a few time steps to ensure it doesn't fail
    #     exp.run(1, 5)

    def test_bay_bridge_toll(self):
        """Verifies that examples/sumo/bay_bridge_toll.py is working."""
        # import the experiment variable from the example
        exp = bay_bridge_toll_example(render=False)

        # run the experiment for a few time steps to ensure it doesn't fail
        exp.run(1, 5)

    def test_minicity(self):
        """Verifies that examples/sumo/minicity.py is working."""
        # import the experiment variable from the example
        exp = minicity_example(render=False)

        # run the experiment for a few time steps to ensure it doesn't fail
        exp.run(1, 5)


class TestRllibExamples(unittest.TestCase):
    """Tests the example scripts in examples/sumo.

    This is done by running each experiment in that folder for five time-steps
    and confirming that it completes one rollout with two workers.
    # FIXME(ev) this test adds several minutes to the testing scheme
    """

    def test_coop_merge(self):
        alg_run, env_name, config = coop_setup()
        self.run_exp(alg_run, env_name, config)

    def test_figure_eight(self):
        alg_run, env_name, config = figure_eight_setup()
        self.run_exp(alg_run, env_name, config)

    def test_green_wave(self):
        alg_run, env_name, config = green_wave_setup()
        self.run_exp(alg_run, env_name, config)

    def test_stabilizing_highway(self):
        alg_run, env_name, config = highway_setup()
        self.run_exp(alg_run, env_name, config)

    def test_ring(self):
        alg_run, env_name, config = ring_setup()
        self.run_exp(alg_run, env_name, config)

    def test_bottleneck(self):
        alg_run, env_name, config = bottleneck_setup()
        self.run_exp(alg_run, env_name, config)

    def test_multi_figure_eight(self):
        alg_run, env_name, config = multi_figure_eight_setup()
        self.run_exp(alg_run, env_name, config)

    def test_multi_ring(self):
        alg_run, env_name, config = multi_ring_setup()
        self.run_exp(alg_run, env_name, config)

    def run_exp(self, alg_run, env_name, config):
        try:
            ray.init(num_cpus=1)
        except Exception:
            pass
        config['train_batch_size'] = 50
        config['horizon'] = 50
        config['sample_batch_size'] = 50
        config['num_workers'] = 0
        config['sgd_minibatch_size'] = 32

        run_experiments({
            'test': {
                'run': alg_run,
                'env': env_name,
                'config': {
                    **config
                },

                'checkpoint_freq': 1,
                'stop': {
                    'training_iteration': 1,
                },
            }
        })


if __name__ == '__main__':
    ray.init(num_cpus=1)  # , redis_address="localhost:6379")
    unittest.main()
