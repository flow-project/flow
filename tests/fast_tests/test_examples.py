import os
import unittest

import ray
from ray.tune import run_experiments

from flow.core.experiment import Experiment

from examples.exp_configs.single_agent.figure_eight import flow_params as single_agent_figure_eight
# from examples.exp_configs.single_agent.green_wave import flow_params as single_agent_green_wave
from examples.exp_configs.single_agent.stabilizing_highway import flow_params as single_agent_stabilizing_highway
from examples.exp_configs.single_agent.stabilizing_the_ring import flow_params as single_agent_stabilizing_the_ring
from examples.exp_configs.single_agent.velocity_bottleneck import flow_params as single_agent_velocity_bottleneck

from examples.exp_configs.multiagent.multiagent_figure_eight import flow_params as multiagent_figure_eight
from examples.exp_configs.multiagent.multiagent_stabilizing_the_ring import \
    flow_params as multiagent_stabilizing_the_ring
# from examples.exp_configs.multiagent.multiagent_traffic_light_grid import setup_exps_PPO as multi_grid_setup
# from examples.exp_configs.multiagent.multiagent_traffic_light_grid import \
#     make_flow_params as multi_grid_setup_flow_params
from examples.exp_configs.multiagent.multiagent_highway import flow_params as multiagent_highway

from examples.train_stable_baselines import run_model as run_stable_baselines_model
from examples.train_rllib import setup_exps as setup_rllib_exps

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
        """Verify that examples/exp_configs/non_rl/bottlenecks.py is working."""
        self.run_simulation(non_rl_bottlenecks)

    def test_figure_eight(self):
        """Verify that examples/exp_configs/non_rl/figure_eight.py is working."""
        self.run_simulation(non_rl_figure_eight)

    def test_grid(self):
        """Verify that examples/exp_configs/non_rl/grid.py is working."""
        self.run_simulation(non_rl_grid)

    def test_highway(self):
        """Verify that examples/exp_configs/non_rl/highway.py is working."""
        # import the experiment variable from the example
        self.run_simulation(non_rl_highway)

    def test_highway_ramps(self):
        """Verify that examples/exp_configs/non_rl/highway_ramps.py is working."""
        self.run_simulation(non_rl_highway_ramps)

    def test_merge(self):
        """Verify that examples/exp_configs/non_rl/merge.py is working."""
        self.run_simulation(non_rl_merge)

    def test_sugiyama(self):
        """Verify that examples/exp_configs/non_rl/sugiyama.py is working."""
        self.run_simulation(non_rl_sugiyama)

    def test_bay_bridge(self):
        """Verify that examples/exp_configs/non_rl/bay_bridge.py is working."""
        # test without inflows and traffic lights
        self.run_simulation(non_rl_bay_bridge)

        # test with inflows
        # FIXME

        # test with traffic lights
        # FIXME

    def test_bay_bridge_toll(self):
        """Verify that examples/exp_configs/non_rl/bay_bridge_toll.py is working."""
        self.run_simulation(non_rl_bay_bridge_toll)

    def test_minicity(self):
        """Verify that examples/exp_configs/non_rl/minicity.py is working."""
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
    @staticmethod
    def run_exp(flow_params):
        run_stable_baselines_model(flow_params, 2, 5, 5)

    def test_figure_eight(self):
        self.run_exp(single_agent_figure_eight)

    def test_green_wave(self):
        pass  # FIXME

    def test_green_wave_inflows(self):
        pass  # FIXME

    def test_stabilizing_highway(self):
        self.run_exp(single_agent_stabilizing_highway)

    def test_ring(self):
        self.run_exp(single_agent_stabilizing_the_ring)

    def test_bottleneck(self):
        self.run_exp(single_agent_velocity_bottleneck)


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
        self.run_exp(single_agent_figure_eight)

    def test_green_wave(self):
        pass  # FIXME

    def test_green_wave_inflows(self):
        pass  # FIXME

    def test_stabilizing_highway(self):
        self.run_exp(single_agent_stabilizing_highway)

    def test_ring(self):
        self.run_exp(single_agent_stabilizing_the_ring)

    def test_bottleneck(self):
        self.run_exp(single_agent_velocity_bottleneck)

    def test_multi_figure_eight(self):
        from examples.exp_configs.multiagent.multiagent_figure_eight import POLICY_GRAPHS
        from examples.exp_configs.multiagent.multiagent_figure_eight import policy_mapping_fn

        kwargs = {
            "policy_graphs": POLICY_GRAPHS,
            "policy_mapping_fn": policy_mapping_fn
        }
        self.run_exp(multiagent_figure_eight, **kwargs)

    def test_multi_ring(self):
        from examples.exp_configs.multiagent.multiagent_stabilizing_the_ring import POLICY_GRAPHS
        from examples.exp_configs.multiagent.multiagent_stabilizing_the_ring import POLICIES_TO_TRAIN
        from examples.exp_configs.multiagent.multiagent_stabilizing_the_ring import policy_mapping_fn

        kwargs = {
            "policy_graphs": POLICY_GRAPHS,
            "policies_to_train": POLICIES_TO_TRAIN,
            "policy_mapping_fn": policy_mapping_fn
        }
        self.run_exp(multiagent_stabilizing_the_ring, **kwargs)

    def test_multi_grid(self):
        pass  # FIXME

    def test_multi_highway(self):
        from examples.exp_configs.multiagent.multiagent_highway import POLICY_GRAPHS
        from examples.exp_configs.multiagent.multiagent_highway import POLICIES_TO_TRAIN
        from examples.exp_configs.multiagent.multiagent_highway import policy_mapping_fn

        kwargs = {
            "policy_graphs": POLICY_GRAPHS,
            "policies_to_train": POLICIES_TO_TRAIN,
            "policy_mapping_fn": policy_mapping_fn
        }
        self.run_exp(multiagent_highway, **kwargs)

    @staticmethod
    def run_exp(flow_params, **kwargs):
        alg_run, env_name, config = setup_rllib_exps(flow_params, 1, 1, **kwargs)

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
