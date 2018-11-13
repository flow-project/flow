"""
Unit tests for Ray
"""

import os
import unittest

import ray
import ray.rllib.agents.ppo as ppo
import ray.tune.registry as registry

from examples.rllib.stabilizing_the_ring import make_create_env

from flow.scenarios.loop import LoopScenario
from flow.controllers.rlcontroller import RLController
from flow.controllers.car_following_models import IDMController
from flow.controllers.routing_controllers import ContinuousRouter

# Inner ring distances closest to the merge are range 300-365 (normalized)
fn_choose_subpolicy = """
def choose_policy(inputs):
    return tf.cast(inputs[:, 0] > 1e6, tf.int32)
"""

HORIZON = 10

BROKEN_TESTS = os.environ.get('BROKEN_TESTS', False)


@unittest.skipUnless(BROKEN_TESTS, "broken test (known issue)")
class TestRay(unittest.TestCase):
    # def setUp(self):
    #     # reload modules, required upon repeated ray.init()

    def test_ray(self):
        """
        Integration test for ray/rllib + flow
        """

        # Test 1: test_two_level_ray
        config = ppo.DEFAULT_CONFIG.copy()
        num_workers = 1
        ray.init(num_cpus=num_workers, redirect_output=False)
        config["num_workers"] = num_workers
        config["timesteps_per_batch"] = min(HORIZON * num_workers, 128)
        config["num_sgd_iter"] = 1
        config["model"].update({"fcnet_hiddens": [3, 3]})
        config["gamma"] = 0.999
        config["min_steps_per_task"] = HORIZON
        config["horizon"] = HORIZON
        config["sgd_batchsize"] = 4

        additional_env_params = {
            "target_velocity": 8,
            "scenario_type": LoopScenario
        }
        additional_net_params = {
            "length": 260,
            "lanes": 1,
            "speed_limit": 30,
            "resolution": 40
        }
        vehicle_params = [
            dict(
                veh_id="rl",
                num_vehicles=1,
                acceleration_controller=(RLController, {}),
                routing_controller=(ContinuousRouter, {})),
            dict(
                veh_id="idm",
                num_vehicles=21,
                acceleration_controller=(IDMController, {}),
                routing_controller=(ContinuousRouter, {}))
        ]

        flow_params = dict(
            sumo=dict(sim_step=0.1, no_step_log=False),
            env=dict(horizon=HORIZON, additional_params=additional_env_params),
            net=dict(
                no_internal_links=False,
                additional_params=additional_net_params),
            veh=vehicle_params,
            initial=dict(spacing="uniform", bunching=30, min_gap=0))

        flow_env_name = "WaveAttenuationPOEnv"
        create_env, env_name = make_create_env(flow_env_name, flow_params, 0)

        # Register as rllib env
        registry.register_env(env_name, create_env)

        alg = ppo.PPOAgent(
            env=env_name, registry=registry.get_registry(), config=config)
        for i in range(1):
            alg.train()
            checkpoint_path = alg.save()
            self.assertTrue("%s.index" % os.path.exists(checkpoint_path))

        # Test 2: test_two_level_ray
        # Integration test for two-level fcnet policy
        # FIXME(cathywu) ray restart currently not supported, so need to tie
        # integration tests together for the time being.
        # reload(ppo)
        # reload(registry)
        config = ppo.DEFAULT_CONFIG.copy()
        num_workers = 1
        # ray.init(num_cpus=num_workers, redirect_output=True)
        config["num_workers"] = num_workers
        config["timesteps_per_batch"] = min(HORIZON * num_workers, 128)
        config["num_sgd_iter"] = 1
        config["model"].update({"fcnet_hiddens": [3, 3]})
        config["gamma"] = 0.999
        config["min_steps_per_task"] = HORIZON
        config["horizon"] = HORIZON
        config["sgd_batchsize"] = 4

        config["model"].update({"fcnet_hiddens": [5, 3]}, )
        options = {
            "num_subpolicies": 2,
            "fn_choose_subpolicy": fn_choose_subpolicy,
            "hierarchical_fcnet_hiddens": [[3, 3]] * 2
        }
        config["model"].update({"custom_options": options})

    def tearDown(self):
        ray.worker.cleanup()


if __name__ == '__main__':
    os.environ["TEST_FLAG"] = "True"
    unittest.main()
