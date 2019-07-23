"""Example of modified minicity network with human-driven vehicles."""
from flow.controllers import IDMController
from flow.controllers import RLController
from flow.core.experiment import Experiment
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig
from flow.core.params import SumoCarFollowingParams, SumoLaneChangeParams
from flow.core.params import VehicleParams
from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.scenarios.minicity import MiniCityScenario
from flow.controllers.routing_controllers import MinicityRouter
import numpy as np

np.random.seed(204)


def minicity_example(render=None,
                     save_render=None,
                     sight_radius=None,
                     pxpm=None,
                     show_radius=None):
    """Perform a simulation of modified minicity of University of Delaware.

    Parameters
    ----------
    render: bool, optional
        specifies whether to use the gui during execution

    Returns
    -------
    exp: flow.core.experiment.Experiment
        A non-rl experiment demonstrating the performance of human-driven
        vehicles on the minicity scenario.
    """
    sim_params = SumoParams(sim_step=0.25)

    # update sim_params values if provided as inputs
    sim_params.render = render or sim_params.render
    sim_params.save_render = save_render or sim_params.save_render
    sim_params.sight_radius = sight_radius or sim_params.sight_radius
    sim_params.pxpm = pxpm or sim_params.pxpm
    sim_params.show_radius = show_radius or sim_params.show_radius

    vehicles = VehicleParams()
    vehicles.add(
        veh_id="idm",
        acceleration_controller=(IDMController, {}),
        routing_controller=(MinicityRouter, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode=1,
        ),
        lane_change_params=SumoLaneChangeParams(
            lane_change_mode="no_lat_collide",
        ),
        initial_speed=0,
        num_vehicles=90)
    vehicles.add(
        veh_id="rl",
        acceleration_controller=(RLController, {}),
        routing_controller=(MinicityRouter, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode="obey_safe_speed",
        ),
        initial_speed=0,
        num_vehicles=10)

    env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

    net_params = NetParams(no_internal_links=False)

    initial_config = InitialConfig(
        spacing="random",
        min_gap=5
    )
    scenario = MiniCityScenario(
        name="minicity",
        vehicles=vehicles,
        initial_config=initial_config,
        net_params=net_params)

    env = AccelEnv(env_params, sim_params, scenario)

    return Experiment(env)


if __name__ == "__main__":
    # import the experiment variable
    # There are six modes of pyglet rendering:
    # No rendering: minicity_example(render=False)
    # SUMO-GUI rendering: minicity_example(render=True)
    # Static grayscale rendering: minicity_example(render="gray")
    # Dynamic grayscale rendering: minicity_example(render="dgray")
    # Static RGB rendering: minicity_example(render="rgb")
    # Dynamic RGB rendering: minicity_example(render="drgb")
    exp = minicity_example(render='drgb',
                           save_render=False,
                           sight_radius=30,
                           pxpm=3,
                           show_radius=True)

    # run for a set number of rollouts / time steps
    exp.run(1, 750)
