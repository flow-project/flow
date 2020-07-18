"""Load an already existing Aimsun template and run the simulation."""
from flow.core.params import AimsunParams, EnvParams, NetParams
from flow.core.params import VehicleParams
from flow.core.params import InFlows
import flow.config as config
from flow.envs import TestEnv
from flow.networks import Network
import os


# no vehicles in the network
vehicles = VehicleParams()

# path to the imported Aimsun template
template_path = os.path.join(config.PROJECT_PATH,
                             "flow/utils/aimsun/small_template.ang")


flow_params = dict(
    # name of the experiment
    exp_tag='aimsun_small_template',

    # name of the flow environment the experiment is running on
    env_name=TestEnv,

    # name of the network class the experiment is running on
    network=Network,

    # simulator that is used by the experiment
    simulator='aimsun',

    # Aimsun-related parameters
    sim=AimsunParams(
        sim_step=0.1,
        render=True,
        emission_path='data',
        replication_name="Replication 930",
        centroid_config_name="Centroid Configuration 910"
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=3000,
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=InFlows(),
        template=template_path
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,
)
