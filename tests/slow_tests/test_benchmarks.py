import unittest
import os
from os import listdir
from os.path import isfile, join
import json
import shutil

import ray
import ray.rllib.agents.ppo as ppo

try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from flow.core.util import ensure_dir

import flow.benchmarks.bottleneck0 as bottleneck0
import flow.benchmarks.bottleneck1 as bottleneck1
import flow.benchmarks.bottleneck2 as bottleneck2
import flow.benchmarks.figureeight0 as figureeight0
import flow.benchmarks.figureeight1 as figureeight1
import flow.benchmarks.figureeight2 as figureeight2
import flow.benchmarks.grid0 as grid0
import flow.benchmarks.grid1 as grid1
import flow.benchmarks.merge0 as merge0
import flow.benchmarks.merge1 as merge1
import flow.benchmarks.merge2 as merge2

from flow.benchmarks.stable_baselines.trpo_runner import run_model
from flow.benchmarks.stable_baselines.trpo_runner import parse_args
from flow.benchmarks.stable_baselines.trpo_runner import save_model

import random

# This removes the randomness in this test
random.seed(a=10)

N_CPUS = 1
ray.init(num_cpus=N_CPUS)

os.environ['TEST_FLAG'] = 'True'


class TestBenchmarks(unittest.TestCase):

    """
    Tests that the baselines in the benchmarks folder are running and
    returning expected values (i.e. values that match those in the CoRL paper
    reported on the website, or other).
    """

    def setUp(self):
        if not os.path.exists('./benchmark_tmp'):
            os.mkdir('benchmark_tmp')

    def tearDown(self):
        shutil.rmtree('benchmark_tmp')

    def ray_runner(self, num_runs, flow_params, version):
        alg_run = 'PPO'
        HORIZON = 10

        agent_cls = get_agent_class(alg_run)
        config = agent_cls._default_config.copy()
        config['num_workers'] = 1
        config['sample_batch_size'] = 50  # arbitrary
        config['train_batch_size'] = 50  # arbitrary
        config['sgd_minibatch_size'] = 10
        config['num_sgd_iter'] = 1
        config['horizon'] = HORIZON

        # save the flow params for replay
        flow_json = json.dumps(
            flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
        config['env_config']['flow_params'] = flow_json
        config['env_config']['run'] = alg_run

        create_env, env_name = make_create_env(params=flow_params,
                                               version=version)

        # Register as rllib env
        register_env(env_name, create_env)

        alg = ppo.PPOAgent(
            env=env_name, config=config)

        for i in range(num_runs):
            alg.train()
            checkpoint_path = alg.save('benchmark_tmp')
            self.assertTrue('%s.index' % os.path.exists(checkpoint_path))

    def test_bottleneck0(self):
        """
        Tests flow/benchmark/baselines/bottleneck0.py
        env_name='BottleneckDesiredVelocityEnv',
        """
        # run the bottleneck to make sure it runs
        self.ray_runner(1, bottleneck0.flow_params, 0)

        # TODO: check that the performance measure is within some range

    def test_bottleneck1(self):
        """
        Tests flow/benchmark/baselines/bottleneck1.py
        """
        # run the bottleneck to make sure it runs
        self.ray_runner(1, bottleneck1.flow_params, 1)

        # TODO: check that the performance measure is within some range

    def test_bottleneck2(self):
        """s
        Tests flow/benchmark/baselines/bottleneck2.py
        """
        # run the bottleneck to make sure it runs
        self.ray_runner(1, bottleneck2.flow_params, 2)

        # TODO: check that the performance measure is within some range

    def test_figure_eight0(self):
        """
        Tests flow/benchmark/baselines/figureeight{0,1,2}.py
        env_name='AccelEnv',
        """
        # run the bottleneck to make sure it runs
        self.ray_runner(1, figureeight0.flow_params, 0)

        # TODO: check that the performance measure is within some range

    def test_figure_eight1(self):
        """
        Tests flow/benchmark/baselines/figureeight{0,1,2}.py
        """
        # run the bottleneck to make sure it runs
        self.ray_runner(1, figureeight1.flow_params, 1)

        # TODO: check that the performance measure is within some range

    def test_figure_eight2(self):
        """
        Tests flow/benchmark/baselines/figureeight{0,1,2}.py
        """
        # run the bottleneck to make sure it runs
        self.ray_runner(1, figureeight2.flow_params, 2)

        # TODO: check that the performance measure is within some range

    def test_grid0(self):
        """
        Tests flow/benchmark/baselines/grid0.py
        env_name='TrafficLightGridPOEnv',
        """
        # run the bottleneck to make sure it runs
        self.ray_runner(1, grid0.flow_params, 0)

        # TODO: check that the performance measure is within some range

    def test_grid1(self):
        """
        Tests flow/benchmark/baselines/grid1.py
        """
        # run the bottleneck to make sure it runs
        self.ray_runner(1, grid1.flow_params, 1)

        # TODO: check that the performance measure is within some range

    def test_merge0(self):
        """
        Tests flow/benchmark/baselines/merge{0,1,2}.py
        env_name='MergePOEnv',
        """
        # run the bottleneck to make sure it runs
        self.ray_runner(1, merge0.flow_params, 0)

        # TODO: check that the performance measure is within some range

    def test_merge1(self):
        """
        Tests flow/benchmark/baselines/merge{0,1,2}.py
        """
        # run the bottleneck to make sure it runs
        self.ray_runner(1, merge1.flow_params, 1)

        # TODO: check that the performance measure is within some range

    def test_merge2(self):
        """
        Tests flow/benchmark/baselines/merge{0,1,2}.py
        """
        # run the bottleneck to make sure it runs
        self.ray_runner(1, merge2.flow_params, 2)


class TestTRPORunner(unittest.TestCase):

    def test_parse_args(self):
        # test the base case
        flags = parse_args(['figureeight0'])
        self.assertEqual(flags.benchmark_name, 'figureeight0')
        self.assertEqual(flags.num_steps, 9e6)
        self.assertEqual(flags.rollout_size, 3e4)
        self.assertEqual(flags.num_cpus, 1)

        # test num_cpus
        flags = parse_args(['figureeight0', '--num_cpus', '3'])
        self.assertEqual(flags.num_cpus, 3)

        # test rollout_size
        flags = parse_args(['figureeight0', '--rollout_size', '2'])
        self.assertEqual(flags.rollout_size, 2)

        # test num_steps
        flags = parse_args(['figureeight0', '--num_steps', '1'])
        self.assertEqual(flags.num_steps, 1)

    def test_trpo_runner(self):
        # test run_model on figure eight 0
        model = run_model(figureeight0.flow_params, 5, 5)

        # test save model
        ensure_dir("./baseline_results")
        save_model(model, figureeight0.flow_params, "./baseline_results")
        files = sorted([f for f in listdir("./baseline_results")
                        if isfile(join("./baseline_results", f))])
        self.assertListEqual(files, ['flow_params.json', 'model.pkl'])

        # delete the generated files
        shutil.rmtree('./baseline_results')


if __name__ == '__main__':
    unittest.main()
