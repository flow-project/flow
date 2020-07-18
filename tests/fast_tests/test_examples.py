from copy import deepcopy
import os
import unittest
import random

import ray
from ray.tune import run_experiments

from flow.core.experiment import Experiment

from examples.exp_configs.rl.singleagent.singleagent_figure_eight import flow_params as singleagent_figure_eight
from examples.exp_configs.rl.singleagent.singleagent_traffic_light_grid import \
    flow_params as singleagent_traffic_light_grid
from examples.exp_configs.rl.singleagent.singleagent_merge import flow_params as singleagent_merge
from examples.exp_configs.rl.singleagent.singleagent_ring import flow_params as singleagent_ring
from examples.exp_configs.rl.singleagent.singleagent_bottleneck import flow_params as singleagent_bottleneck

from examples.exp_configs.rl.multiagent.adversarial_figure_eight import flow_params as adversarial_figure_eight
from examples.exp_configs.rl.multiagent.multiagent_i210 import flow_params as multiagent_i210
from examples.exp_configs.rl.multiagent.multiagent_figure_eight import flow_params as multiagent_figure_eight
from examples.exp_configs.rl.multiagent.multiagent_merge import flow_params as multiagent_merge
from examples.exp_configs.rl.multiagent.lord_of_the_rings import \
    flow_params as lord_of_the_rings
from examples.exp_configs.rl.multiagent.multiagent_ring import flow_params as multiagent_ring
from examples.exp_configs.rl.multiagent.multiagent_traffic_light_grid import \
    flow_params as multiagent_traffic_light_grid
from examples.exp_configs.rl.multiagent.multiagent_highway import flow_params as multiagent_highway

from examples.simulate import parse_args as parse_simulate_args
from examples.train import parse_args as parse_train_args
from examples.train import run_model_stablebaseline as run_stable_baselines_model
from examples.train import setup_exps_rllib as setup_rllib_exps
from examples.train import train_h_baselines

from examples.exp_configs.non_rl.bay_bridge import flow_params as non_rl_bay_bridge
from examples.exp_configs.non_rl.bay_bridge_toll import flow_params as non_rl_bay_bridge_toll
from examples.exp_configs.non_rl.bottleneck import flow_params as non_rl_bottleneck
from examples.exp_configs.non_rl.figure_eight import flow_params as non_rl_figure_eight
from examples.exp_configs.non_rl.traffic_light_grid import flow_params as non_rl_traffic_light_grid
from examples.exp_configs.non_rl.highway import flow_params as non_rl_highway
from examples.exp_configs.non_rl.highway_ramps import flow_params as non_rl_highway_ramps
from examples.exp_configs.non_rl.merge import flow_params as non_rl_merge
from examples.exp_configs.non_rl.minicity import flow_params as non_rl_minicity
from examples.exp_configs.non_rl.ring import flow_params as non_rl_ring
from examples.exp_configs.non_rl.i210_subnetwork import flow_params as non_rl_i210
from examples.exp_configs.non_rl.highway_single import flow_params as non_rl_highway_single

os.environ['TEST_FLAG'] = 'True'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# This removes the randomness in this test
random.seed(a=10)


class TestNonRLExamples(unittest.TestCase):
    """Tests the experiment configurations in examples/exp_configs/non_rl.

    This is done by running an experiment form of each config for a
    few time steps. Note that, this does not test for any refactoring changes
    done to the functions within the experiment class.
    """

    def test_parse_args(self):
        """Validate the functionality of the parse_args method in simulate.py."""
        # test the default case
        args = parse_simulate_args(["exp_config"])

        self.assertDictEqual(vars(args), {
            'aimsun': False,
            'exp_config': 'exp_config',
            'gen_emission': False,
            'no_render': False,
            'num_runs': 1
        })

        # test the case when optional args are specified
        args = parse_simulate_args([
            "exp_config",
            '--aimsun',
            '--gen_emission',
            '--no_render',
            '--num_runs', '2'
        ])

        self.assertDictEqual(vars(args), {
            'aimsun': True,
            'exp_config': 'exp_config',
            'gen_emission': True,
            'no_render': True,
            'num_runs': 2
        })

    def test_bottleneck(self):
        """Verify that examples/exp_configs/non_rl/bottleneck.py is working."""
        self.run_simulation(non_rl_bottleneck)

    def test_figure_eight(self):
        """Verify that examples/exp_configs/non_rl/figure_eight.py is working."""
        self.run_simulation(non_rl_figure_eight)

    def test_traffic_light_grid(self):
        """Verify that examples/exp_configs/non_rl/traffic_light_grid.py is working."""
        self.run_simulation(non_rl_traffic_light_grid)

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

    def test_ring(self):
        """Verify that examples/exp_configs/non_rl/ring.py is working."""
        self.run_simulation(non_rl_ring)

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

    def test_i210(self):
        """Verify that examples/exp_configs/non_rl/i210_subnetwork.py is working."""
        self.run_simulation(non_rl_i210)

    def test_highway_single(self):
        """Verify that examples/exp_configs/non_rl/highway_single.py is working."""
        self.run_simulation(non_rl_highway_single)

    @staticmethod
    def run_simulation(flow_params):
        # make the horizon small and set render to False
        flow_params['sim'].render = False
        flow_params['env'].horizon = 5

        # create an experiment object
        exp = Experiment(flow_params)

        # run the experiment for one run
        exp.run(1)


class TestTrain(unittest.TestCase):

    def test_parse_args(self):
        """Tests the parse_args method in train.py."""
        # test the default case
        args = parse_train_args(["exp_config"])

        self.assertDictEqual(vars(args), {
            'exp_config': 'exp_config',
            'rl_trainer': 'rllib',
            'num_cpus': 1,
            'num_steps': 5000,
            'rollout_size': 1000,
            'checkpoint_path': None
        })

        # test the case when optional args are specified
        args = parse_train_args([
            "exp_config",
            "--rl_trainer", "h-baselines",
            "--num_cpus" "2",
            "--num_steps", "3",
            "--rollout_size", "4",
            "--checkpoint_path", "5",
        ])

        self.assertDictEqual(vars(args), {
            'checkpoint_path': '5',
            'exp_config': 'exp_config',
            'num_cpus': 1,
            'num_steps': 3,
            'rl_trainer': 'h-baselines',
            'rollout_size': 4
        })


class TestStableBaselineExamples(unittest.TestCase):
    """Tests the example scripts in examples/exp_configs/rl/singleagent for stable_baselines.

    This is done by running each experiment in that folder for five time-steps
    and confirming that it completes one rollout with two workers.
    """
    @staticmethod
    def run_exp(flow_params):
        train_model = run_stable_baselines_model(flow_params, 1, 4, 4)
        train_model.env.close()

    def test_singleagent_figure_eight(self):
        self.run_exp(singleagent_figure_eight)

    def test_singleagent_traffic_light_grid(self):
        self.run_exp(singleagent_traffic_light_grid)

    def test_singleagent_merge(self):
        self.run_exp(singleagent_merge)

    def test_singleagent_ring(self):
        self.run_exp(singleagent_ring)

    def test_singleagent_bottleneck(self):
        self.run_exp(singleagent_bottleneck)


class TestHBaselineExamples(unittest.TestCase):
    """Tests the functionality of the h-baselines features in train.py.

    This is done by running a set of experiments for 10 time-steps and
    confirming that it runs.
    """
    @staticmethod
    def run_exp(env_name, multiagent):
        train_h_baselines(
            env_name=env_name,
            args=[
                env_name,
                "--initial_exploration_steps", "1",
                "--total_steps", "10"
            ],
            multiagent=multiagent,
        )

    def test_singleagent_ring(self):
        self.run_exp("singleagent_ring", multiagent=False)

    def test_multiagent_ring(self):
        self.run_exp("multiagent_ring", multiagent=True)


class TestRllibExamples(unittest.TestCase):
    """Tests the example scripts in examples/exp_configs/rl/singleagent and
    examples/exp_configs/rl/multiagent for RLlib.

    This is done by running each experiment in that folder for five time-steps
    and confirming that it completes one rollout with two workers.
    # FIXME(ev) this test adds several minutes to the testing scheme
    """
    def setUp(self):
        if not ray.is_initialized():
            ray.init(num_cpus=1)

    def test_singleagent_figure_eight(self):
        self.run_exp(singleagent_figure_eight)

    def test_singleagent_traffic_light_grid(self):
        self.run_exp(singleagent_traffic_light_grid)

    def test_singleagent_traffic_light_grid_inflows(self):
        pass  # FIXME

    def test_singleagent_merge(self):
        self.run_exp(singleagent_merge)

    def test_singleagent_ring(self):
        self.run_exp(singleagent_ring)

    def test_singleagent_bottleneck(self):
        self.run_exp(singleagent_bottleneck)

    def test_adversarial_figure_eight(self):
        from examples.exp_configs.rl.multiagent.adversarial_figure_eight import POLICY_GRAPHS as af8pg
        from examples.exp_configs.rl.multiagent.adversarial_figure_eight import policy_mapping_fn as af8pmf

        kwargs = {
            "policy_graphs": af8pg,
            "policy_mapping_fn": af8pmf
        }
        self.run_exp(adversarial_figure_eight, **kwargs)

    def test_multiagent_figure_eight(self):
        from examples.exp_configs.rl.multiagent.multiagent_figure_eight import POLICY_GRAPHS as mf8pg
        from examples.exp_configs.rl.multiagent.multiagent_figure_eight import policy_mapping_fn as mf8pmf

        kwargs = {
            "policy_graphs": mf8pg,
            "policy_mapping_fn": mf8pmf
        }
        self.run_exp(multiagent_figure_eight, **kwargs)

    def test_lord_of_the_rings(self):
        from examples.exp_configs.rl.multiagent.lord_of_the_rings import POLICY_GRAPHS as ltrpg
        from examples.exp_configs.rl.multiagent.lord_of_the_rings import POLICIES_TO_TRAIN as ltrpt
        from examples.exp_configs.rl.multiagent.lord_of_the_rings import policy_mapping_fn as ltrpmf

        kwargs = {
            "policy_graphs": ltrpg,
            "policies_to_train": ltrpt,
            "policy_mapping_fn": ltrpmf
        }
        self.run_exp(lord_of_the_rings, **kwargs)

    def test_multiagent_ring(self):
        from examples.exp_configs.rl.multiagent.multiagent_ring import POLICY_GRAPHS as mrpg
        from examples.exp_configs.rl.multiagent.multiagent_ring import policy_mapping_fn as mrpmf

        kwargs = {
            "policy_graphs": mrpg,
            "policy_mapping_fn": mrpmf
        }
        self.run_exp(multiagent_ring, **kwargs)

    def test_multiagent_merge(self):
        from examples.exp_configs.rl.multiagent.multiagent_merge import POLICY_GRAPHS as mmpg
        from examples.exp_configs.rl.multiagent.multiagent_merge import policy_mapping_fn as mmpmf

        kwargs = {
            "policy_graphs": mmpg,
            "policy_mapping_fn": mmpmf
        }
        self.run_exp(multiagent_merge, **kwargs)

    def test_multi_traffic_light_grid(self):
        from examples.exp_configs.rl.multiagent.multiagent_traffic_light_grid import POLICY_GRAPHS as mtlpg
        from examples.exp_configs.rl.multiagent.multiagent_traffic_light_grid import POLICIES_TO_TRAIN as mtlpt
        from examples.exp_configs.rl.multiagent.multiagent_traffic_light_grid import policy_mapping_fn as mtlpmf

        kwargs = {
            "policy_graphs": mtlpg,
            "policies_to_train": mtlpt,
            "policy_mapping_fn": mtlpmf
        }
        self.run_exp(multiagent_traffic_light_grid, **kwargs)

    def test_multi_highway(self):
        from examples.exp_configs.rl.multiagent.multiagent_highway import POLICY_GRAPHS as mhpg
        from examples.exp_configs.rl.multiagent.multiagent_highway import POLICIES_TO_TRAIN as mhpt
        from examples.exp_configs.rl.multiagent.multiagent_highway import policy_mapping_fn as mhpmf

        kwargs = {
            "policy_graphs": mhpg,
            "policies_to_train": mhpt,
            "policy_mapping_fn": mhpmf
        }
        self.run_exp(multiagent_highway, **kwargs)

    def test_multiagent_i210(self):
        from examples.exp_configs.rl.multiagent.multiagent_i210 import POLICIES_TO_TRAIN as mi210pr
        from examples.exp_configs.rl.multiagent.multiagent_i210 import policy_mapping_fn as mi210mf

        from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
        from ray.tune.registry import register_env
        from flow.utils.registry import make_create_env
        # test observation space 1
        flow_params = deepcopy(multiagent_i210)
        flow_params['env'].additional_params['lead_obs'] = True
        create_env, env_name = make_create_env(params=flow_params, version=0)

        # register as rllib env
        register_env(env_name, create_env)

        # multiagent configuration
        test_env = create_env()
        obs_space = test_env.observation_space
        act_space = test_env.action_space

        POLICY_GRAPHS = {'av': (PPOTFPolicy, obs_space, act_space, {})}

        kwargs = {
            "policy_graphs": POLICY_GRAPHS,
            "policies_to_train": mi210pr,
            "policy_mapping_fn": mi210mf
        }
        self.run_exp(flow_params, **kwargs)

        # test observation space 2
        flow_params = deepcopy(multiagent_i210)
        flow_params['env'].additional_params['lead_obs'] = False
        create_env, env_name = make_create_env(params=flow_params, version=0)

        # register as rllib env
        register_env(env_name, create_env)

        # multiagent configuration
        test_env = create_env()
        obs_space = test_env.observation_space
        act_space = test_env.action_space

        POLICY_GRAPHS = {'av': (PPOTFPolicy, obs_space, act_space, {})}

        kwargs = {
            "policy_graphs": POLICY_GRAPHS,
            "policies_to_train": mi210pr,
            "policy_mapping_fn": mi210mf
        }
        self.run_exp(flow_params, **kwargs)

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
