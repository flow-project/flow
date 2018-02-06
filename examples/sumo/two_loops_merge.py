"""
Example of ring road with larger merging ring.
"""
import logging

from flow.core.experiment import SumoExperiment
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.vehicles import Vehicles

from flow.controllers.car_following_models import IDMController
from flow.controllers.routing_controllers import ContinuousRouter

from flow.envs.two_loops_one_merging import TwoLoopsMergeEnv
from flow.scenarios.two_loops_one_merging.gen import TwoLoopOneMergingGenerator
from flow.scenarios.two_loops_one_merging.two_loops_one_merging_scenario \
    import TwoLoopsOneMergingScenario, ADDITIONAL_NET_PARAMS


def two_loops_merge_example(sumo_binary=None):
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
                 routing_controller=(ContinuousRouter, {}),
                 num_vehicles=12)
    vehicles.add(veh_id="merge-idm",
                 acceleration_controller=(IDMController, {}),
                 routing_controller=(ContinuousRouter, {}),
                 num_vehicles=5)

    additional_env_params = {"target_velocity": 8}
    env_params = EnvParams(max_decel=6, max_accel=3,
                           additional_params=additional_env_params)

    additional_net_params = ADDITIONAL_NET_PARAMS.copy()
    net_params = NetParams(
        no_internal_links=False,
        additional_params=additional_net_params
    )

    initial_config = InitialConfig(
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
    exp = two_loops_merge_example()

    # run for a set number of rollouts / time steps
    exp.run(1, 1500)
