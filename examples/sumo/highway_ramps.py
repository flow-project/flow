"""Example of a highway section network with on/off ramps."""

from flow.core.params import SumoParams, EnvParams, \
    NetParams, InitialConfig, InFlows, SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.core.experiment import Experiment
from flow.scenarios.highway_ramps import HighwayRampsScenario, \
                                         ADDITIONAL_NET_PARAMS
from flow.controllers import IDMController
from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS


additional_net_params = ADDITIONAL_NET_PARAMS.copy()

# lengths
additional_net_params["highway_length"] = 600
additional_net_params["on_ramps_length"] = 65
additional_net_params["off_ramps_length"] = 65

# number of lanes
additional_net_params["highway_lanes"] = 4
additional_net_params["on_ramps_lanes"] = 1
additional_net_params["off_ramps_lanes"] = 1

# speed limits
additional_net_params["highway_speed"] = 15
additional_net_params["on_ramps_speed"] = 10
additional_net_params["off_ramps_speed"] = 10

# ramps
additional_net_params["on_ramps_pos"] = [200, 400]
additional_net_params["off_ramps_pos"] = [300, 500]

# probability of exiting at the next off-ramp
additional_net_params["next_off_ramp_proba"] = 0.05

# inflow rates in vehs/hour
HIGHWAY_INFLOW_RATE = 1000
ON_RAMPS_INFLOW_RATE = 200


def highway_ramps_example(render=True):
    """
    Perform a simulation of vehicles on a highway section with ramps.

    Parameters
    ----------
    render: bool, optional
        Specifies whether or not to use the GUI during the simulation.

    Returns
    -------
    exp: flow.core.experiment.Experiment
        A non-RL experiment demonstrating the performance of human-driven
        vehicles on a highway section with on and off ramps.
    """
    sim_params = SumoParams(
        render=True,
        emission_path="./data/",
        sim_step=0.2,
        restart_instance=True)

    vehicles = VehicleParams()
    vehicles.add(
        veh_id="human",
        acceleration_controller=(IDMController, {
            "noise": 0.2
        }),
        car_following_params=SumoCarFollowingParams(
            speed_mode="obey_safe_speed",
        ))

    env_params = EnvParams(
        additional_params=ADDITIONAL_ENV_PARAMS,
        sims_per_step=5,
        warmup_steps=0)

    inflows = InFlows()
    inflows.add(
        veh_type="human",
        edge="highway_0",
        vehs_per_hour=HIGHWAY_INFLOW_RATE,
        depart_lane="free",
        depart_speed="speedLimit",
        name="highway_flow")
    for i in range(len(additional_net_params["on_ramps_pos"])):
        inflows.add(
            veh_type="human",
            edge="on_ramp_{}".format(i),
            vehs_per_hour=ON_RAMPS_INFLOW_RATE,
            depart_lane="first",
            depart_speed="speedLimit",
            name="on_ramp_flow")

    net_params = NetParams(
        inflows=inflows,
        additional_params=additional_net_params)

    initial_config = InitialConfig(spacing="uniform", perturbation=5.0)

    scenario = HighwayRampsScenario(
        name="highway-ramp",
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config)

    env = AccelEnv(env_params, sim_params, scenario)

    return Experiment(env)


if __name__ == "__main__":
    exp = highway_ramps_example()
    exp.run(1, 3600, convert_to_csv=True)
