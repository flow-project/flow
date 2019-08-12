"""Load an already existing Aimsun template and run the simulation."""

from flow.core.experiment import Experiment
from flow.core.params import AimsunParams, EnvParams, NetParams
from flow.core.params import VehicleParams
from flow.envs import TestEnv
from flow.scenarios.loop import Scenario
from flow.core.params import InFlows
import flow.config as config
import os

sim_params = AimsunParams(
    sim_step=0.1,
    render=True,
    emission_path='data',
    replication_name="Replication 930",
    centroid_config_name="Centroid Configuration 910")

env_params = EnvParams()
vehicles = VehicleParams()

template_path = os.path.join(config.PROJECT_PATH,
                             "flow/utils/aimsun/small_template.ang")

scenario = Scenario(
    name="aimsun_small_template",
    vehicles=vehicles,
    net_params=NetParams(
        inflows=InFlows(),
        template=template_path
    )
)

env = TestEnv(env_params, sim_params, scenario, simulator='aimsun')
exp = Experiment(env)
exp.run(1, 3000)
