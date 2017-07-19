import unittest
import logging

from cistar.core.exp import SumoExperiment
from cistar.envs.loop_accel import SimpleAccelerationEnvironment
from cistar.scenarios.loop.loop_scenario import LoopScenario
from cistar.controllers.car_following_models import *
from cistar.controllers.lane_change_controllers import *
from cistar.controllers.rlcontroller import RLController

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite, stub
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy



class TestTwoLaneTwoController(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.WARNING)

        self.sumo_params = {"time_step":0.1, "human_sm": 1, "rl_sm": 1, 
                "human_lc": "strategic", "rl_lc": "no_lat_collide"}
        self.sumo_binary = "sumo"
        self.type_params = {"rl":(5, (RLController, {}), None, 0),
               "cfm":(3, (IDMController, {}), None, 0)}
        self.env_params = {"target_velocity": 8, "max-deacc":3, "max-acc":3}
        self.net_params = {"length": 200, "lanes": 2, "speed_limit": 35,
                           "resolution": 40, "net_path": "tests/debug/net/"}
        self.cfg_params = {"start_time": 0, "end_time": 1000,
                           "cfg_path": "tests/debug/cfg/"}

    def test_it_runs(self):
        stub(globals())
        scenario = LoopScenario("test-two-lane-two-controller",
                                self.type_params, self.net_params,
                                self.cfg_params)

        # FIXME(cathywu) it currently looks like there's no lane changing,
        # although there should be.
        # self.sumo_binary = "sumo-gui"
        env = SimpleAccelerationEnvironment(self.env_params,
                             self.sumo_binary, self.sumo_params, scenario)

        logging.info("Experiment Set Up complete")

        print("experiment initialized")

        env = normalize(env)

        for seed in [1]: # [1, 5, 10, 73, 56]
            policy = GaussianMLPPolicy(
                env_spec=env.spec,
                hidden_sizes=(32,32)
            )

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            algo = TRPO(
                env=env,
                policy=policy,
                baseline=baseline,
                batch_size=1000,
                max_path_length=100,
                # whole_paths=True,
                n_itr=2,
                # discount=0.99,
                # step_size=0.01,
            )
            # algo.train()

            run_experiment_lite(
                algo.train(),
                # Number of parallel workers for sampling
                n_parallel=1,
                # Only keep the snapshot parameters for the last iteration
                snapshot_mode="last",
                # Specifies the seed for the experiment. If this is not provided, a random seed
                # will be used
                seed=seed,
                mode="local",
                exp_prefix="leah-test-exp"
                # plot=True,
            )

        env.terminate()


if __name__ == '__main__':
    unittest.main()
