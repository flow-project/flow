import os
import unittest

import ray
from ray.tune import run_experiments

from examples.rllib.figure_eight import setup_exps as figure_eight_setup
from examples.rllib.traffic_light_grid import setup_exps as traffic_light_grid_setup
from examples.rllib.stabilizing_highway import setup_exps as highway_setup
from examples.rllib.stabilizing_the_ring import setup_exps as ring_setup
from examples.rllib.velocity_bottleneck import setup_exps as bottleneck_setup
from examples.rllib.multiagent_exps.multiagent_figure_eight \
   import setup_exps as multi_figure_eight_setup
from examples.rllib.multiagent_exps.multiagent_stabilizing_the_ring \
    import setup_exps as multi_ring_setup
from examples.rllib.multiagent_exps.multiagent_traffic_light_grid \
    import setup_exps_PPO as multi_grid_setup
from examples.rllib.multiagent_exps.multiagent_traffic_light_grid \
    import make_flow_params as multi_grid_setup_flow_params
from examples.rllib.multiagent_exps.multiagent_highway import flow_params \
    as multi_highway_flow_params
from examples.rllib.multiagent_exps.multiagent_highway import setup_exps \
    as multi_highway_setup

from examples.stable_baselines.figure_eight import run_model as run_figure_eight
from examples.stable_baselines.traffic_light_grid import run_model as run_traffic_light_grid
from examples.stable_baselines.stabilizing_highway import run_model as run_stabilizing_highway
from examples.stable_baselines.stabilizing_the_ring import run_model as run_stabilizing_ring
from examples.stable_baselines.velocity_bottleneck import run_model as run_velocity_bottleneck

from examples.sumo.bay_bridge import bay_bridge_example
from examples.sumo.bay_bridge_toll import bay_bridge_toll_example
from examples.sumo.bottlenecks import bottleneck_example
from examples.sumo.density_exp import run_bottleneck
from examples.sumo.figure_eight import figure_eight_example
from examples.sumo.traffic_light_grid import traffic_light_grid_example
from examples.sumo.highway import highway_example
from examples.sumo.highway_ramps import highway_ramps_example
from examples.sumo.merge import merge_example
from examples.sumo.minicity import minicity_example
from examples.sumo.sugiyama import sugiyama_example

os.environ['TEST_FLAG'] = 'True'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


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

    def test_traffic_light_grid(self):
        """Verifies that examples/sumo/traffic_light_grid.py is working."""
        # test the example in the absence of inflows
        exp = traffic_light_grid_example(render=False, use_inflows=False)
        exp.run(1, 5)

        # test the example in the presence of inflows
        exp = traffic_light_grid_example(render=False, use_inflows=True)
        exp.run(1, 5)

    def test_highway(self):
        """Verifies that examples/sumo/highway.py is working."""
        # import the experiment variable from the example
        exp = highway_example(render=False)

        # run the experiment for a few time steps to ensure it doesn't fail
        exp.run(1, 5)

    def test_highway_ramps(self):
        """Verifies that examples/sumo/highway_ramps.py is working."""
        # import the experiment variable from the example
        exp = highway_ramps_example(render=False)

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

    def test_bay_bridge(self):
        """Verifies that examples/sumo/bay_bridge.py is working."""
        # import the experiment variable from the example
        exp = bay_bridge_example(render=False)

        # run the experiment for a few time steps to ensure it doesn't fail
        exp.run(1, 5)

        # import the experiment variable from the example with inflows
        exp = bay_bridge_example(render=False, use_inflows=True)

        # run the experiment for a few time steps to ensure it doesn't fail
        exp.run(1, 5)

        # import the experiment variable from the example with traffic lights
        exp = bay_bridge_example(render=False, use_traffic_lights=True)

        # run the experiment for a few time steps to ensure it doesn't fail
        exp.run(1, 5)

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

    def test_density_exp(self):
        """Verifies that examples/sumo/density_exp.py is working."""
        run_bottleneck.remote(100, 1, 10, render=False)


class TestStableBaselineExamples(unittest.TestCase):
    """Tests the example scripts in examples/stable_baselines.

        This is done by running each experiment in that folder for five time-steps
        and confirming that it completes one rollout with two workers.
    """
    def test_run_traffic_light_grid(self):
        run_traffic_light_grid(num_steps=5)

    def test_run_figure_eight(self):
        run_figure_eight(num_steps=5)

    def test_run_stabilizing_highway(self):
        run_stabilizing_highway(num_steps=5)

    def test_run_stabilizing_ring(self):
        run_stabilizing_ring(num_steps=5)

    def test_run_velocity_bottleneck(self):
        run_velocity_bottleneck(num_steps=5)


class TestRllibExamples(unittest.TestCase):
    """Tests the example scripts in examples/rllib.

    This is done by running each experiment in that folder for five time-steps
    and confirming that it completes one rollout with two workers.
    # FIXME(ev) this test adds several minutes to the testing scheme
    """
    def setUp(self):
        if not ray.is_initialized():
            ray.init(num_cpus=1)

    def test_figure_eight(self):
        alg_run, env_name, config = figure_eight_setup()
        self.run_exp(alg_run, env_name, config)

    def test_traffic_light_grid(self):
        # test the example in the absence of inflows
        alg_run, env_name, config = traffic_light_grid_setup(use_inflows=False)
        self.run_exp(alg_run, env_name, config)

    def test_traffic_light_grid_inflows(self):
        # test the example in the presence of inflows
        alg_run, env_name, config = traffic_light_grid_setup(use_inflows=True)
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

    def test_multi_grid(self):
        flow_params = multi_grid_setup_flow_params(1, 1, 300)
        alg_run, env_name, config = multi_grid_setup(flow_params)
        self.run_exp(alg_run, env_name, config)

    def test_multi_highway(self):
        flow_params = multi_highway_flow_params
        alg_run, env_name, config = multi_highway_setup(flow_params)
        self.run_exp(alg_run, env_name, config)

    @staticmethod
    def run_exp(alg_run, env_name, config):
        try:
            ray.init(num_cpus=1)
        except Exception as e:
            print("ERROR", e)
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
    try:
        ray.init(num_cpus=1)
    except Exception as e:
        print("ERROR", e)
    unittest.main()
    ray.shutdown()
