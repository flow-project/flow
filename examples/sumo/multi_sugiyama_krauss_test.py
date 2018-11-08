"""Used as an example of sugiyama experiment.

This example consists of 22 IDM cars on a ring creating shockwaves.
"""

from flow.controllers import SumoCarFollowingController, SafeAggressiveLaneChanger, ContinuousRouter
from flow.core.experiment import SumoExperiment
from flow.core.params import SumoParams, EnvParams, \
    InitialConfig, NetParams, SumoCarFollowingParams, SumoLaneChangeParams
from flow.core.vehicles import Vehicles
from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.scenarios.loop import LoopScenario, ADDITIONAL_NET_PARAMS


def sugiyama_example(render=None):
    """
    Perform a simulation of vehicles on a ring road.

    Parameters
    ----------
    render : bool, optional
        specifies whether to use sumo's gui during execution

    Returns
    -------
    exp: flow.core.SumoExperiment type
        A non-rl experiment demonstrating the performance of human-driven
        vehicles on a ring road.
    """
    sumo_params = SumoParams(sim_step=0.1, render=True)

    if render is not None:
        sumo_params.render = render

    vehicles = Vehicles()
    vehicles.add(
        veh_id="krauss_fast",
        acceleration_controller=(SumoCarFollowingController, {}),
        sumo_car_following_params=SumoCarFollowingParams(car_follow_model="Krauss", speedDev=0.7,),
        lane_change_controller=(SafeAggressiveLaneChanger, {"target_velocity": 22.25, "threshold": 0.8}),
        sumo_lc_params=SumoLaneChangeParams(lcKeepRight=0, lcAssertive=0.5,
                                       lcSpeedGain=1.5, lcSpeedGainRight=1.0),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=80)

    env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS, starting_position_shuffle=True)

    additional_net_params = ADDITIONAL_NET_PARAMS.copy()
    additional_net_params["lanes"] = 4
    additional_net_params["length"] = 400
    net_params = NetParams(additional_params=additional_net_params)

    initial_config = InitialConfig(bunching=40, spacing="uniform")

    scenario = LoopScenario(
        name="sugiyama",
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config)

    env = AccelEnv(env_params, sumo_params, scenario)

    return SumoExperiment(env, scenario)


if __name__ == "__main__":
    # import the experiment variable
    exp = sugiyama_example()

    # run for a set number of rollouts / time steps
    exp.run(1, 3000)
