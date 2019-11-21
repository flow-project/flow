"""Load an already existing Aimsun template and run the simulation."""

from flow.core.experiment import Experiment
from flow.core.params import AimsunParams, EnvParams, NetParams
from flow.core.params import VehicleParams
from flow.envs import TestEnv
from flow.networks import Network
from flow.core.params import InFlows
import flow.config as config
import os

# Setting parameters
SIM_STEP = 0.8  # seconds
REPLICATION_NAME = # "Replication (one hour)"
CENTROID_CONF_NAME = # "Centroid Configuration 8040652"
SIM_LOCATION = # "example/aimsun/toyota/base_case1.ang"
NETWORK_NAME = # "toyota_base_case1"
RENDER = True


sim_params = AimsunParams(sim_step=SIM_STEP,
                          render=RENDER,
                          restart_instance=False,
                          emission_path='data',
                          replication_name=REPLICATION_NAME, 
                          centroid_config_name=CENTROID_CONF_NAME)
    

env_params = EnvParams()
env_params = EnvParams(horizon=HORIZON,
                       warmup_steps=int(np.ceil(120/detector_step)),
                       sims_per_step=int(detector_step/sim_step),
                       additional_params=ADDITIONAL_ENV_PARAMS)


vehicles = VehicleParams()

template_path = os.path.join(config.PROJECT_PATH,
                             SIM_LOCATION) 

network = Network(
    name=NETWORK_NAME, 
    vehicles=vehicles,
    net_params=NetParams(
        inflows=InFlows(),
        template=template_path
    )
)

env = TestEnv(env_params, sim_params, network, simulator='aimsun')
exp = Experiment(env)
exp.run(1, 3000)