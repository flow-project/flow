"""
Example of ring road with larger merging ring.
"""
import logging

from flow.core.experiment import SumoExperiment
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    SumoCarFollowingParams, SumoLaneChangeParams
from flow.core.vehicles import Vehicles

from flow.controllers.car_following_models import IDMController
from flow.controllers.lane_change_controllers import SumoLaneChangeController
from flow.controllers.routing_controllers import ContinuousRouter

from flow.envs.two_loops_one_merging import TwoLoopsMergeEnv
from flow.scenarios.two_loops_one_merging_new.gen import \
    TwoLoopOneMergingGenerator
from flow.scenarios.two_loops_one_merging_new.scenario import \
    TwoLoopsOneMergingScenario


def two_loops_merge_straight_example(sumo_binary=None):
    logging.basicConfig(level=logging.INFO)

    sumo_params = SumoParams(sim_step=0.1, emission_path="./data/",
                             sumo_binary="sumo-gui")

    if sumo_binary is not None:
        sumo_params.sumo_binary = sumo_binary

    # note that the vehicles are added sequentially by the generator,
    # so place the merging vehicles after the vehicles in the ring
    vehicles = Vehicles()
    vehicles.add(veh_id="idm",
                 acceleration_controller=(IDMController, {}),
                 lane_change_controller=(SumoLaneChangeController, {}),
                 routing_controller=(ContinuousRouter, {}),
                 num_vehicles=7,
                 sumo_car_following_params=SumoCarFollowingParams(
                     minGap=0.0, tau=0.5),
                 sumo_lc_params=SumoLaneChangeParams())
    vehicles.add(veh_id="merge-idm",
                 acceleration_controller=(IDMController, {}),
                 lane_change_controller=(SumoLaneChangeController, {}),
                 routing_controller=(ContinuousRouter, {}),
                 num_vehicles=7,
                 sumo_car_following_params=SumoCarFollowingParams(
                     minGap=0.01, tau=0.5),
                 sumo_lc_params=SumoLaneChangeParams())

    additional_env_params = {"target_velocity": 20}
    env_params = EnvParams(additional_params=additional_env_params)

    additional_net_params = {"ring_radius": 75,
                             "inner_lanes": 1,
                             "outer_lanes": 1,
                             "lane_length": 75,
                             "speed_limit": 30,
                             "resolution": 40}
    net_params = NetParams(
        no_internal_links=False,
        additional_params=additional_net_params
    )

    initial_config = InitialConfig(
        x0=50,
        spacing="custom",
        additional_params={"merge_bunching": 0}
    )

    scenario = TwoLoopsOneMergingScenario(
        name="two-loop-one-merging",
        generator_class=TwoLoopOneMergingGenerator,
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config
    )

    env = TwoLoopsMergeEnv(env_params, sumo_params, scenario)

    exp = SumoExperiment(env, scenario)

    logging.info("Experiment Set Up complete")

    return exp


if __name__ == "__main__":
    # import the experiment variable
    exp = two_loops_merge_straight_example()

    # run for a set number of rollouts / time steps
    exp.run(1, 1500, convert_to_csv=True)
