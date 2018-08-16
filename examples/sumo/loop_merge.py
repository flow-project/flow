"""
Example of ring road with larger merging ring.
"""

from flow.controllers import IDMController, SumoLaneChangeController, \
    ContinuousRouter
from flow.core.experiment import SumoExperiment
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    SumoCarFollowingParams, SumoLaneChangeParams
from flow.core.vehicles import Vehicles
from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.scenarios.loop_merge.scenario import \
    TwoLoopsOneMergingScenario, ADDITIONAL_NET_PARAMS
from flow.scenarios.loop_merge.gen import TwoLoopOneMergingGenerator


def loop_merge_example(sumo_binary=None):
    sumo_params = SumoParams(sim_step=0.1,
                             emission_path="./data/",
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
                 speed_mode="no_collide",
                 sumo_car_following_params=SumoCarFollowingParams(
                     minGap=0.0, tau=0.5),
                 sumo_lc_params=SumoLaneChangeParams())
    vehicles.add(veh_id="merge-idm",
                 acceleration_controller=(IDMController, {}),
                 lane_change_controller=(SumoLaneChangeController, {}),
                 routing_controller=(ContinuousRouter, {}),
                 num_vehicles=10,
                 speed_mode="no_collide",
                 sumo_car_following_params=SumoCarFollowingParams(
                     minGap=0.01, tau=0.5),
                 sumo_lc_params=SumoLaneChangeParams())

    env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

    additional_net_params = ADDITIONAL_NET_PARAMS.copy()
    additional_net_params["ring_radius"] = 50
    additional_net_params["inner_lanes"] = 1
    additional_net_params["outer_lanes"] = 1
    additional_net_params["lane_length"] = 75
    net_params = NetParams(
        no_internal_links=False,
        additional_params=additional_net_params
    )

    initial_config = InitialConfig(
        x0=50,
        spacing="uniform",
        additional_params={"merge_bunching": 0}
    )

    scenario = TwoLoopsOneMergingScenario(
        name="two-loop-one-merging",
        generator_class=TwoLoopOneMergingGenerator,
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config
    )

    env = AccelEnv(env_params, sumo_params, scenario)

    return SumoExperiment(env, scenario)


if __name__ == "__main__":
    # import the experiment variable
    exp = loop_merge_example()

    # run for a set number of rollouts / time steps
    exp.run(1, 1500, convert_to_csv=True)
