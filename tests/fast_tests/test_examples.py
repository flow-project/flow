import os
import unittest

import ray
from ray.tune import run_experiments

from flow.core.experiment import Experiment

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
from examples.exp_configs.multiagent.multiagent_traffic_light_grid \
    import setup_exps_PPO as multi_grid_setup
from examples.exp_configs.multiagent.multiagent_traffic_light_grid \
    import make_flow_params as multi_grid_setup_flow_params
from examples.exp_configs.multiagent.multiagent_highway import flow_params \
    as multi_highway_flow_params
from examples.exp_configs.multiagent.multiagent_highway import setup_exps \
    as multi_highway_setup

from examples.stable_baselines.figure_eight import run_model as run_figure_eight
from examples.stable_baselines.green_wave import run_model as run_green_wave
from examples.stable_baselines.stabilizing_highway import run_model as run_stabilizing_highway
from examples.stable_baselines.stabilizing_the_ring import run_model as run_stabilizing_ring
from examples.stable_baselines.velocity_bottleneck import run_model as run_velocity_bottleneck

from examples.exp_configs.non_rl.bay_bridge import flow_params as non_rl_bay_bridge
from examples.exp_configs.non_rl.bay_bridge_toll import flow_params as non_rl_bay_bridge_toll
from examples.exp_configs.non_rl.bottlenecks import flow_params as non_rl_bottlenecks
from examples.exp_configs.non_rl.figure_eight import flow_params as non_rl_figure_eight
from examples.exp_configs.non_rl.grid import flow_params as non_rl_grid
from examples.exp_configs.non_rl.highway import flow_params as non_rl_highway
from examples.exp_configs.non_rl.highway_ramps import flow_params as non_rl_highway_ramps
from examples.exp_configs.non_rl.merge import flow_params as non_rl_merge
from examples.exp_configs.non_rl.minicity import flow_params as non_rl_minicity
from examples.exp_configs.non_rl.sugiyama import flow_params as non_rl_sugiyama

os.environ['TEST_FLAG'] = 'True'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class TestNonRLExamples(unittest.TestCase):
    """Tests the experiment configurations in examples/exp_configs/non_rl.

    This is done by running an experiment form of each config for a
    few time steps. Note that, this does not test for any refactoring changes
    done to the functions within the experiment class.
    """

    def test_bottleneck(self):
        """Verifies that examples/exp_configs/non_rl/bottlenecks.py is working."""
        self.run_simulation(non_rl_bottlenecks)

    def test_figure_eight(self):
        """Verifies that examples/exp_configs/non_rl/figure_eight.py is working."""
        self.run_simulation(non_rl_figure_eight)

    def test_grid(self):
        """Verifies that examples/exp_configs/non_rl/grid.py is working."""
        self.run_simulation(non_rl_grid)

    def test_highway(self):
        """Verifies that examples/exp_configs/non_rl/highway.py is working."""
        # import the experiment variable from the example
        self.run_simulation(non_rl_highway)

    def test_highway_ramps(self):
        """Verifies that examples/exp_configs/non_rl/highway_ramps.py is working."""
        self.run_simulation(non_rl_highway_ramps)

    def test_merge(self):
        """Verifies that examples/exp_configs/non_rl/merge.py is working."""
        self.run_simulation(non_rl_merge)

    def test_sugiyama(self):
        """Verifies that examples/exp_configs/non_rl/sugiyama.py is working."""
        self.run_simulation(non_rl_sugiyama)

    def test_bay_bridge(self):
        """Verifies that examples/exp_configs/non_rl/bay_bridge.py is working."""
        # test without inflows and traffic lights
        self.run_simulation(non_rl_bay_bridge)

        # test with inflows
        # FIXME

        # test with traffic lights
        # FIXME

    def test_bay_bridge_toll(self):
        """Verifies that examples/exp_configs/non_rl/bay_bridge_toll.py is working."""
        self.run_simulation(non_rl_bay_bridge_toll)

    def test_minicity(self):
        """Verifies that examples/exp_configs/non_rl/minicity.py is working."""
        self.run_simulation(non_rl_minicity)

    @staticmethod
    def run_simulation(flow_params):
        # make the horizon small and set render to False
        flow_params['sim'].render = False
        flow_params['env'].horizon = 5

        # create an experiment object
        exp = Experiment(flow_params)

        # run the experiment for one run
        exp.run(1)


class TestStableBaselineExamples(unittest.TestCase):
    """Tests the example scripts in examples/stable_baselines.

        This is done by running each experiment in that folder for five time-steps
        and confirming that it completes one rollout with two workers.
    """
    def test_run_green_wave(self):
        run_green_wave(num_steps=5)

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

    def test_coop_merge(self):
        alg_run, env_name, config = coop_setup()
        self.run_exp(alg_run, env_name, config)

    def test_figure_eight(self):
        alg_run, env_name, config = figure_eight_setup()
        self.run_exp(alg_run, env_name, config)

    def test_green_wave(self):
        # test the example in the absence of inflows
        alg_run, env_name, config = green_wave_setup(use_inflows=False)
        self.run_exp(alg_run, env_name, config)

    def test_green_wave_inflows(self):
        # test the example in the presence of inflows
        alg_run, env_name, config = green_wave_setup(use_inflows=True)
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
