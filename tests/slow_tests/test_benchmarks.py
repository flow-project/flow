import unittest
import os
import json

import ray
import ray.rllib.agents.ppo as ppo

from ray.rllib.agents.agent import get_agent_class
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder


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

N_CPUS = 1
ray.init(num_cpus=N_CPUS, redirect_output=True)

os.environ["TEST_FLAG"] = "True"


class TestBenchmarks(unittest.TestCase):

    """
    Tests that the baselines in the benchmarks folder are running and
    returning expected values (i.e. values that match those in the CoRL paper
    reported on the website, or other).
    """

    """
    self.env, self.scenario = setup_bottlenecks()
        self.exp = SumoExperiment(self.env, self.scenario)
        self.exp.run(5, 50)
    """

    def ray_runner(self, num_runs, flow_params, version):
        alg_run = "PPO"
        HORIZON = 10
        N_ROLLOUTS = 1

        agent_cls = get_agent_class(alg_run)
        config = agent_cls._default_config.copy()
        config["num_workers"] = 1
        config["timesteps_per_batch"] = HORIZON * N_ROLLOUTS
        config["gamma"] = 0.999  # discount rate
        config["model"].update({"fcnet_hiddens": [100, 50, 25]})
        config["use_gae"] = True
        config["lambda"] = 0.97
        config["kl_target"] = 0.02
        config["num_sgd_iter"] = 1
        config["horizon"] = HORIZON
        config["observation_filter"] = "NoFilter"

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
            checkpoint_path = alg.save()
            self.assertTrue("%s.index" % os.path.exists(checkpoint_path))

    def test_bottleneck0(self):
        """
        Tests flow/benchmark/baselines/bottleneck0.py
        env_name="DesiredVelocityEnv",
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
        """
        Tests flow/benchmark/baselines/bottleneck2.py
        """
        # run the bottleneck to make sure it runs
        self.ray_runner(1, bottleneck2.flow_params, 2)

        # TODO: check that the performance measure is within some range

    def test_figure_eight0(self):
        """
        Tests flow/benchmark/baselines/figureeight{0,1,2}.py
        env_name="AccelEnv",
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
        env_name="PO_TrafficLightGridEnv",
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
        env_name="WaveAttenuationMergePOEnv",
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

        # TODO: check that the performance measure is within some range

    # create_env, env_name


if __name__ == '__main__':
    unittest.main()
