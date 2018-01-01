from importlib import reload
import os
import unittest

import ray
import ray.rllib.ppo as ppo
import ray.tune.registry as registry

from examples.rllib.stabilizing_the_ring import make_create_env


class TestRay(unittest.TestCase):
    # def setUp(self):
    #     # reload modules, required upon repeated ray.init()

    def test_ray(self):
        """
        Integration test for ray/rllib + flow
        """
        # Test 1: test_two_level_ray
        config = ppo.DEFAULT_CONFIG.copy()
        horizon = 500
        num_workers = 3
        ray.init(num_cpus=num_workers, redirect_output=True)
        config["num_workers"] = num_workers
        config["timesteps_per_batch"] = horizon * num_workers
        config["num_sgd_iter"] = 2
        config["model"].update({"fcnet_hiddens": [3, 3]})
        config["gamma"] = 0.999
        config["horizon"] = horizon

        flow_env_name = "WaveAttenuationPOEnv"
        create_env, env_name = make_create_env(flow_env_name, 0)

        # Register as rllib env
        registry.register_env(env_name, create_env)

        alg = ppo.PPOAgent(env=env_name, registry=registry.get_registry(),
                           config=config)
        for i in range(2):
            alg.train()
            checkpoint_path = alg.save()
            self.assertTrue("%s.index" % os.path.exists(checkpoint_path))

        # Test 2: test_two_level_ray
        # Integration test for two-level fcnet policy
        # FIXME(cathywu) ray restart currently not supported, so need to tie
        # integration tests together for the time being.
        # reload(ppo)
        # reload(registry)
        import cloudpickle
        config = ppo.DEFAULT_CONFIG.copy()
        horizon = 500
        num_workers = 3
        # ray.init(num_cpus=num_workers, redirect_output=True)
        config["num_workers"] = num_workers
        config["timesteps_per_batch"] = horizon * num_workers
        config["num_sgd_iter"] = 2
        config["gamma"] = 0.999
        config["horizon"] = horizon

        config["model"].update(
            {"fcnet_hiddens": [[5, 3]] * 2})
        config["model"]["user_data"] = {}
        config["model"]["user_data"].update({"num_subpolicies": 2,
                                             "fn_choose_subpolicy": list(
                                                 cloudpickle.dumps(lambda x: 0))})

        flow_env_name = "WaveAttenuationPOEnv"
        create_env, env_name = make_create_env(flow_env_name, 1)

        # Register as rllib env
        registry.register_env(env_name, create_env)

        alg = ppo.PPOAgent(env=env_name, registry=registry.get_registry(),
                           config=config)
        for i in range(1):
            alg.train()

    def tearDown(self):
        ray.worker.cleanup()


if __name__ == '__main__':
    unittest.main()
