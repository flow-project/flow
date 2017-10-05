import unittest
import logging

from flow.core.experiment import SumoExperiment
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.vehicles import Vehicles

from flow.controllers.routing_controllers import ContinuousRouter
from flow.controllers.car_following_models import *
from flow.controllers.rlcontroller import RLController

from flow.envs.loop_accel import SimpleAccelerationEnvironment
from flow.scenarios.loop.gen import CircleGenerator
from flow.scenarios.loop.loop_scenario import LoopScenario

from rllab.envs.normalized_env import normalize
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.gym_env import GymEnv

class SingleLaneOneController(unittest.TestCase):
    """
    Tests IDM Controller, Continuous Router, & SimpleAccelerationEnvironment
    """
    def setUp(self):
        logging.basicConfig(level=logging.WARNING)

        sumo_params = SumoParams()

        vehicles = Vehicles()
        vehicles.add_vehicles("idm", (IDMController, {}), None, (ContinuousRouter, {}), 0, 22)

        additional_env_params = {"target_velocity": 8, "num_steps": 100}
        env_params = EnvParams(additional_params=additional_env_params)

        additional_net_params = {"length": 230, "lanes": 2, "speed_limit": 30, "resolution": 40}
        net_params = NetParams(additional_params=additional_net_params)

        initial_config = InitialConfig()

        scenario = LoopScenario("SingleLaneOneControllerTest", CircleGenerator, vehicles, net_params, initial_config)

        env = SimpleAccelerationEnvironment(env_params, sumo_params, scenario)

        self.exp = SumoExperiment(env, scenario)

    def test_it_runs(self):
        self.exp.run(1, 200)  # params: num_runs, num_steps


class SingleLaneMixedSingleAgentRL(unittest.TestCase):
    """
    Tests IDM Controller, RL Controller, Continuous Router, SimpleAccelerationEnvironment & RlLab functionality
    """
    def setUp(self):

        logging.basicConfig(level=logging.WARNING)

        sumo_params = SumoParams()

        vehicles = Vehicles()
        vehicles.add_vehicles("rl", (RLController, {}), None, (ContinuousRouter, {}), 0, 2)
        vehicles.add_vehicles("idm", (IDMController, {}), None, (ContinuousRouter, {}), 0, 8)

        additional_env_params = {"target_velocity": 8, "num_steps": 100}
        env_params = EnvParams(additional_params=additional_env_params)

        additional_net_params = {"length": 200, "lanes": 2, "speed_limit": 35, "resolution": 40}
        net_params = NetParams(additional_params=additional_net_params)

        initial_config = InitialConfig()

        scenario = LoopScenario("MultiLaneMixedRl", CircleGenerator, vehicles, net_params, initial_config)

        env_name = "SimpleLaneChangingAccelerationEnvironment"
        pass_params = (env_name, sumo_params, vehicles, env_params, net_params,
                       initial_config, scenario)

        env = GymEnv(env_name, record_video=False, register_params=pass_params)
        horizon = env.horizon
        env = normalize(env)
        logging.info("Experiment Set Up complete")

        env = normalize(env)

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(6, 6)
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        self.algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=300,
            max_path_length=horizon,
            # whole_paths=True,
            n_itr=2,
            # discount=0.99,
            # step_size=0.01,
        )


    def test_it_runs(self):
        self.algo.train()

if __name__ == '__main__':
    unittest.main()
