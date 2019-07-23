"""Example of a merge network with human-driven vehicles.

In the absence of autonomous vehicles, the network exhibits properties of
convective instability, with perturbations propagating upstream from the merge
point before exiting the network.
"""

import flow.core.params as params
from flow.core.experiment import Experiment
from flow.scenarios.merge import MergeScenario, ADDITIONAL_NET_PARAMS
from flow.controllers import IDMController
from flow.envs.merge import WaveAttenuationMergePOEnv, ADDITIONAL_ENV_PARAMS

# inflow rate at the highway
HIGHWAY_RATE = 2000
# inflow rate at the on-ramp
MERGE_RATE = 500


def merge_example(render=None):
    """Perform a simulation of vehicles on a merge.

    Parameters
    ----------
    render: bool, optional
        specifies whether to use the gui during execution

    Returns
    -------
    exp: flow.core.experiment.Experiment
        A non-rl experiment demonstrating the performance of human-driven
        vehicles on a merge.
    """
    sim_params = params.AimsunParams(
        render=True,
        emission_path="./data/",
        sim_step=0.5,
        restart_instance=False)

    if render is not None:
        sim_params.render = render

    vehicles = params.VehicleParams()
    vehicles.add(
        veh_id="human",
        acceleration_controller=(IDMController, {
            "noise": 0.2
        }),
        num_vehicles=5)

    env_params = params.EnvParams(
        additional_params=ADDITIONAL_ENV_PARAMS,
        sims_per_step=5,
        warmup_steps=0)

    inflow = params.InFlows()
    inflow.add(
        veh_type="human",
        edge="inflow_highway",
        vehs_per_hour=HIGHWAY_RATE,
        departLane="free",
        departSpeed=10)
    inflow.add(
        veh_type="human",
        edge="inflow_merge",
        vehs_per_hour=MERGE_RATE,
        departLane="free",
        departSpeed=7.5)

    additional_net_params = ADDITIONAL_NET_PARAMS.copy()
    additional_net_params["merge_lanes"] = 1
    additional_net_params["highway_lanes"] = 1
    additional_net_params["pre_merge_length"] = 500
    net_params = params.NetParams(
        inflows=inflow,
        no_internal_links=False,
        additional_params=additional_net_params)

    initial_config = params.InitialConfig()

    scenario = MergeScenario(
        name="merge-baseline",
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config)

    env = WaveAttenuationMergePOEnv(
        env_params, sim_params, scenario, simulator='aimsun')

    return Experiment(env)


if __name__ == "__main__":
    # import the experiment variable
    exp = merge_example()

    # run for a set number of rollouts / time steps
    exp.run(1, 3600)
