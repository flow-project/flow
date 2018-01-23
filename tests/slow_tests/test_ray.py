"""
Unit tests for Ray
"""

from importlib import reload
import os
import unittest

import ray
import ray.rllib.ppo as ppo
import ray.tune.registry as registry

from examples.rllib.stabilizing_the_ring import make_create_env, HORIZON

from flow.scenarios.loop.loop_scenario import LoopScenario
from flow.controllers.rlcarfollowingcontroller import RLCarFollowingController
from flow.controllers.car_following_models import IDMController
from flow.controllers.routing_controllers import ContinuousRouter

# Inner ring distances closest to the merge are range 300-365 (normalized)
fn_choose_subpolicy = """
def choose_policy(inputs):
    return tf.cast(inputs[:, 0] > 1e6, tf.int32)
"""


class TestRay(unittest.TestCase):
    # def setUp(self):
    #     # reload modules, required upon repeated ray.init()

    def test_ray(self):
        """
        Integration test for ray/rllib + flow
        """

        # Test 1: test_two_level_ray
        config = ppo.DEFAULT_CONFIG.copy()
        horizon = 100
        num_workers = 2
        ray.init(num_cpus=num_workers, redirect_output=True)
        config["num_workers"] = num_workers
        config["timesteps_per_batch"] = horizon * num_workers
        config["num_sgd_iter"] = 2
        config["model"].update({"fcnet_hiddens": [3, 3]})
        config["gamma"] = 0.999
        config["horizon"] = horizon

        additional_env_params = {"target_velocity": 8, "max-deacc": -1,
                         "max-acc": 1, "num_steps": horizon,
                         "scenario_type": LoopScenario}
        additional_net_params = {"length": 260, "lanes": 1, "speed_limit": 30,
                                 "resolution": 40}
        vehicle_params = [dict(veh_id="rl", num_vehicles=1,
                               acceleration_controller=(RLCarFollowingController, {}),
                               routing_controller=(ContinuousRouter, {})),
                          dict(veh_id="idm", num_vehicles=21,
                               acceleration_controller=(IDMController, {}),
                               routing_controller=(ContinuousRouter, {}))
                         ]

        flow_params = dict(
                        sumo=dict(sim_step=0.1),
                        env=dict(additional_params=additional_env_params),
                        net=dict(no_internal_links=False,
                            additional_params=additional_net_params),
                        veh=vehicle_params,
                        initial=dict(spacing="uniform", bunching=30, min_gap=0)
                      )

        flow_env_name = "WaveAttenuationPOEnv"
        create_env, env_name = make_create_env(flow_env_name, flow_params, 0)

        # Register as rllib env
        registry.register_env(env_name, create_env)

        alg = ppo.PPOAgent(env=env_name, registry=registry.get_registry(),
                           config=config)
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
        horizon = HORIZON
        num_workers = 2
        # ray.init(num_cpus=num_workers, redirect_output=True)
        config["num_workers"] = num_workers
        config["timesteps_per_batch"] = horizon * num_workers
        config["num_sgd_iter"] = 2
        config["gamma"] = 0.999
        config["horizon"] = horizon

        config["model"].update(
            {"fcnet_hiddens": [5, 3]},)
        options = {"num_subpolicies": 2,
                    "fn_choose_subpolicy": fn_choose_subpolicy,
                    "hierarchical_fcnet_hiddens": [[3, 3]] * 2}
        config["model"].update({"custom_options": options})

        flow_env_name = "WaveAttenuationPOEnv"
        create_env, env_name = make_create_env(flow_env_name, flow_params, 1)

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
