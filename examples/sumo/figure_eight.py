"""Example of a figure 8 network with human-driven vehicles.

Right-of-way dynamics near the intersection causes vehicles to queue up on
either side of the intersection, leading to a significant reduction in the
average speed of vehicles in the network.
"""
from flow.controllers import IDMController, StaticLaneChanger, ContinuousRouter
from flow.core.experiment import Experiment
from flow.core.params import SumoParams, EnvParams, NetParams, \
    SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.scenarios.figure_eight import Figure8Scenario, ADDITIONAL_NET_PARAMS


def figure_eight_example(render=None):
    """
    Perform a simulation of vehicles on a figure eight.

    Parameters
    ----------
    render: bool, optional
        specifies whether to use the gui during execution

    Returns
    -------
    exp: flow.core.experiment.Experiment
        A non-rl experiment demonstrating the performance of human-driven
        vehicles on a figure eight.
    """
    sim_params = SumoParams(render=True)

    if render is not None:
        sim_params.render = render

    vehicles = VehicleParams()
    vehicles.add(
        veh_id="idm",
        acceleration_controller=(IDMController, {}),
        lane_change_controller=(StaticLaneChanger, {}),
        routing_controller=(ContinuousRouter, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode="obey_safe_speed",
            decel=1.5,
        ),
        initial_speed=0,
        num_vehicles=14)

    env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

    additional_net_params = ADDITIONAL_NET_PARAMS.copy()
    net_params = NetParams(
        no_internal_links=False, additional_params=additional_net_params)

    scenario = Figure8Scenario(
        name="figure8",
        vehicles=vehicles,
        net_params=net_params)

    env = AccelEnv(env_params, sim_params, scenario)

    return Experiment(env)


if __name__ == "__main__":
    # import the experiment variable
    exp = figure_eight_example()

    # run for a set number of rollouts / time steps
    exp.run(1, 1500)
