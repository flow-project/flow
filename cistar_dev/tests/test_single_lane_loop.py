import unittest
import logging

from cistar.core.experiment import SumoExperiment
from cistar.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from cistar.core.vehicles import Vehicles

from cistar.controllers.routing_controllers import ContinuousRouter
from cistar.controllers.car_following_models import *
from cistar.controllers.rlcontroller import RLController

from cistar.envs.loop_accel import SimpleAccelerationEnvironment
from cistar.scenarios.loop.gen import CircleGenerator
from cistar.scenarios.loop.loop_scenario import LoopScenario

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

        sumo_params = SumoParams(time_step=0.1, human_speed_mode="aggressive", sumo_binary="sumo")

        vehicles = Vehicles()
        vehicles.add_vehicles("idm", (IDMController, {}), None, (ContinuousRouter, {}), 0, 22)

        additional_env_params = {"target_velocity": 8, "max-deacc": 3, "max-acc": 3, "num_steps": 500}
        env_params = EnvParams(additional_params=additional_env_params)

        additional_net_params = {"length": 230, "lanes": 1, "speed_limit": 30, "resolution": 40}
        net_params = NetParams(additional_params=additional_net_params)

        initial_config = InitialConfig(bunching=20)

        scenario = LoopScenario("SingleLaneOneControllerTest", CircleGenerator, vehicles, net_params, initial_config)

        env = SimpleAccelerationEnvironment(env_params, sumo_params, scenario)

        self.exp = SumoExperiment(env, scenario)

    def test_it_runs(self):
        self.exp.run(1, 1000)  # params: num_runs, num_steps


class SingleLaneMixedSingleAgentRL(unittest.TestCase):
    """
    Tests IDM Controller, RL Controller, Continuous Router, SimpleAccelerationEnvironment & RlLab functionality
    """
    def setUp(self):
        logging.basicConfig(level=logging.WARNING)

        sumo_params = SumoParams(time_step=0.1, human_speed_mode="aggressive", sumo_binary="sumo")

        vehicles = Vehicles()
        vehicles.add_vehicles("idm", (IDMController, {}), None, (ContinuousRouter, {}), 0, 21)
        vehicles.add_vehicles("rl", (RLController, {}), None, (ContinuousRouter, {}), 0, 1)

        additional_env_params = {"target_velocity": 8, "max-deacc": 3, "max-acc": 3, "num_steps": 500}
        env_params = EnvParams(additional_params=additional_env_params)

        additional_net_params = {"length": 230, "lanes": 1, "speed_limit": 30, "resolution": 40}
        net_params = NetParams(additional_params=additional_net_params)

        initial_config = InitialConfig(bunching=20)

        scenario = LoopScenario("SingleLaneMixedRL", CircleGenerator, vehicles, net_params, initial_config)

        env_name = "SimpleAccelerationEnvironment"
        pass_params = (env_name, sumo_params, vehicles, env_params, net_params,
                       initial_config, scenario)

        env = GymEnv(env_name, record_video=False, register_params=pass_params)
        horizon = env.horizon
        env = normalize(env)

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(4,)
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        self.algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=500,
            max_path_length=horizon,
            # whole_paths=True,
            n_itr=1,  # 1000
            # discount=0.99,
            # step_size=0.01,
        )

    def test_it_runs(self):
        self.algo.train()

if __name__ == '__main__':
    unittest.main()
