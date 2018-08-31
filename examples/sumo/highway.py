"""Example of an open multi-lane network with human-driven vehicles."""

from flow.controllers import IDMController
from flow.core.experiment import SumoExperiment
from flow.core.params import SumoParams, EnvParams, \
    NetParams, InitialConfig, InFlows
from flow.core.vehicles import Vehicles
from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.scenarios.highway.gen import HighwayGenerator
from flow.scenarios.highway.scenario import HighwayScenario, \
    ADDITIONAL_NET_PARAMS


def highway_example(sumo_binary=None):
    """
    Perform a simulation of vehicles on a highway.

    Parameters
    ----------
    sumo_binary: bool, optional
        specifies whether to use sumo's gui during execution

    Returns
    -------
    exp: flow.core.SumoExperiment type
        A non-rl experiment demonstrating the performance of human-driven
        vehicles on a figure eight.
    """
    sumo_params = SumoParams(sumo_binary="sumo-gui")

    if sumo_binary is not None:
        sumo_params.sumo_binary = sumo_binary

    vehicles = Vehicles()
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
        edge="highway",
        probability=0.25,
        departLane="free",
        departSpeed=20)
    inflow.add(
        veh_type="human2",
        edge="highway",
        probability=0.25,
        departLane="free",
        departSpeed=20)

    additional_net_params = ADDITIONAL_NET_PARAMS.copy()
    net_params = NetParams(
        in_flows=inflow, additional_params=additional_net_params)

    initial_config = InitialConfig(spacing="uniform", shuffle=True)

    scenario = HighwayScenario(
        name="highway",
        generator_class=HighwayGenerator,
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config)

    env = AccelEnv(env_params, sumo_params, scenario)

    return SumoExperiment(env, scenario)


if __name__ == "__main__":
    # import the experiment variable
    exp = highway_example()

    # run for a set number of rollouts / time steps
    exp.run(1, 1500)
