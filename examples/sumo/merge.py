"""Example of a merge network with human-driven vehicles.

In the absence of autonomous vehicles, the network exhibits properties of
convective instability, with perturbations propagating upstream from the merge
point before exiting the network.
"""

from flow.core.params import SumoParams, EnvParams, \
    NetParams, InitialConfig, InFlows, SumoCarFollowingParams
from flow.core.vehicles import Vehicles
from flow.core.experiment import SumoExperiment
from flow.scenarios.merge.gen import MergeGenerator
from flow.scenarios.merge.scenario import MergeScenario, \
    ADDITIONAL_NET_PARAMS
from flow.controllers import IDMController
from flow.envs.merge import WaveAttenuationMergePOEnv, ADDITIONAL_ENV_PARAMS

# inflow rate at the highway
FLOW_RATE = 2000


def merge_example(render=None):
    """
    Perform a simulation of vehicles on a merge.

    Parameters
    ----------
    render: bool, optional
        specifies whether to use sumo's gui during execution

    Returns
    -------
    exp: flow.core.SumoExperiment type
        A non-rl experiment demonstrating the performance of human-driven
        vehicles on a merge.
    """
    sumo_params = SumoParams(
        render=True,
        emission_path="./data/",
        sim_step=0.2,
        restart_instance=True)

    if render is not None:
        sumo_params.render = render

    vehicles = Vehicles()
    vehicles.add(
        veh_id="human",
        acceleration_controller=(IDMController, {
            "noise": 0.2
        }),
        sumo_car_following_params=SumoCarFollowingParams(
            speed_mode="no_collide",
        ),
        num_vehicles=5)

    env_params = EnvParams(
        additional_params=ADDITIONAL_ENV_PARAMS,
        sims_per_step=5,
        warmup_steps=0)

    inflow = InFlows()
    inflow.add(
        veh_type="human",
        edge="inflow_highway",
        vehs_per_hour=FLOW_RATE,
        departLane="free",
        departSpeed=10)
    inflow.add(
        veh_type="human",
        edge="inflow_merge",
        vehs_per_hour=100,
        departLane="free",
        departSpeed=7.5)

    additional_net_params = ADDITIONAL_NET_PARAMS.copy()
    additional_net_params["merge_lanes"] = 1
    additional_net_params["highway_lanes"] = 1
    additional_net_params["pre_merge_length"] = 500
    net_params = NetParams(
        inflows=inflow,
        no_internal_links=False,
        additional_params=additional_net_params)

    initial_config = InitialConfig(spacing="uniform", perturbation=5.0)

    scenario = MergeScenario(
        name="merge-baseline",
        generator_class=MergeGenerator,
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config)

    env = WaveAttenuationMergePOEnv(env_params, sumo_params, scenario)

    return SumoExperiment(env, scenario)


if __name__ == "__main__":
    # import the experiment variable
    exp = merge_example()

    # run for a set number of rollouts / time steps
    exp.run(1, 3600, convert_to_csv=False)
