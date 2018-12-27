"""Example of an open multi-lane network with human-driven vehicles."""

from flow.controllers import IDMController
from flow.core.experiment import Experiment
from flow.core.params import SumoParams, EnvParams, \
    NetParams, InitialConfig, InFlows
from flow.core.params import VehicleParams
from flow.envs.loop.lane_changing import LaneChangeAccelEnv, \
    ADDITIONAL_ENV_PARAMS
from flow.scenarios.highway import HighwayScenario, ADDITIONAL_NET_PARAMS


def highway_example(render=None):
    """
    Perform a simulation of vehicles on a highway.

    Parameters
    ----------
    render : bool, optional
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
        veh_id="human",
        acceleration_controller=(IDMController, {}),
        num_vehicles=20)
    vehicles.add(
        veh_id="human2",
        acceleration_controller=(IDMController, {}),
        num_vehicles=20)

    env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

    inflow = InFlows()
    inflow.add(
        veh_type="human",
        edge="highway_0",
        probability=0.25,
        departLane="free",
        departSpeed=20)
    inflow.add(
        veh_type="human2",
        edge="highway_0",
        probability=0.25,
        departLane="free",
        departSpeed=20)

    additional_net_params = ADDITIONAL_NET_PARAMS.copy()
    net_params = NetParams(
        inflows=inflow, additional_params=additional_net_params)

    initial_config = InitialConfig(spacing="uniform", shuffle=True)

    scenario = HighwayScenario(
        name="highway",
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config)

    env = LaneChangeAccelEnv(env_params, sim_params, scenario)

    return Experiment(env)


if __name__ == "__main__":
    # import the experiment variable
    exp = highway_example()

    # run for a set number of rollouts / time steps
    exp.run(1, 1500)
