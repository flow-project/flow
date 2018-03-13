"""
Example of ring road with larger merging ring.
"""
import logging

from flow.core.experiment import SumoExperiment
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams,\
    SumoCarFollowingParams, SumoLaneChangeParams
from flow.core.vehicles import Vehicles

from flow.controllers.car_following_models import *
from flow.controllers.lane_change_controllers import *
from flow.controllers.routing_controllers import *

from flow.envs.two_loops_one_merging import TwoLoopsMergeNoRLPOEnv
from flow.scenarios.two_loops_one_merging_new.gen import \
    TwoLoopOneMergingGenerator
from flow.scenarios.two_loops_one_merging_new.scenario import \
    TwoLoopsOneMergingScenario, ADDITIONAL_NET_PARAMS

HORIZON = 1500


def cooperative_merge_example(sumo_binary=None):

    logging.basicConfig(level=logging.INFO)

    sumo_params = SumoParams(sim_step=0.1, sumo_binary="sumo-gui")

    if sumo_binary is not None:
        sumo_params.sumo_binary = sumo_binary

    # note that the vehicles are added sequentially by the generator,
    # so place the merging vehicles after the vehicles in the ring
    vehicles = Vehicles()
    # Inner ring vehicles
    vehicles.add(veh_id="idm",
                 acceleration_controller=(
                     IDMController, {"noise": 0.2}),
                 lane_change_controller=(
                     SumoLaneChangeController, {}),
                 routing_controller=(ContinuousRouter, {}),
                 num_vehicles=7,
                 sumo_car_following_params=SumoCarFollowingParams(
                     minGap=0.0, tau=0.5),
                 sumo_lc_params=SumoLaneChangeParams())
    # Outer ring vehicles
    vehicles.add(veh_id="merge-human",
                 acceleration_controller=(
                     IDMController, {"noise": 0.2}),
                 lane_change_controller=(
                     SumoLaneChangeController, {}),
                 routing_controller=(ContinuousRouter, {}),
                 num_vehicles=10,
                 sumo_car_following_params=SumoCarFollowingParams(
                     minGap=0.0, tau=0.5),
                 sumo_lc_params=SumoLaneChangeParams())

    additional_env_params = {"target_velocity": 20, "num_steps": HORIZON}
    env_params = EnvParams(additional_params=additional_env_params)

    additional_net_params = ADDITIONAL_NET_PARAMS.copy()
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
        name="cooperative_merge_sumo",
        generator_class=TwoLoopOneMergingGenerator,
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config
    )

    env = TwoLoopsMergeNoRLPOEnv(env_params, sumo_params, scenario)

    exp = SumoExperiment(env, scenario)

    logging.info("Experiment Set Up complete")

    return exp


if __name__ == "__main__":

    # import the experiment variable from the example
    exp = cooperative_merge_example()

    # run the experiment for a few time steps to ensure it doesn't fail
    exp.run(1, 1500)
