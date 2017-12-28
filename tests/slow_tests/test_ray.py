import os
import unittest


class TestRay(unittest.TestCase):
    """
    Integration test for ray/rllib
    """

    def runTest(self):
        import ray
        import ray.rllib.ppo as ppo
        from ray.tune.registry import get_registry, register_env as register_rllib_env
        from examples.stabilizing_the_ring_ray import create_env

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
        env_name = flow_env_name+'-v0'

        # Register as rllib env
        register_rllib_env(env_name, create_env)

        alg = ppo.PPOAgent(env=env_name, registry=get_registry(), config=config)
        for i in range(2):
            alg.train()
            checkpoint_path = alg.save()
            self.assertTrue("%s.index" % os.path.exists(checkpoint_path))


if __name__ == '__main__':
    unittest.main()
