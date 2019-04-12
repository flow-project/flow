"""
Load an already existing Aimsun template and run the simulation
"""

from flow.core.experiment import Experiment
from flow.core.params import AimsunParams, EnvParams, NetParams
from flow.core.params import VehicleParams
from flow.envs import TestEnv
from flow.scenarios.loop import Scenario
from flow.controllers.rlcontroller import RLController



sim_params = AimsunParams(
    sim_step=0.1,
    render=True,
    emission_path='data',
    subnetwork_name="Subnetwork 8028981",
    centroid_config_name="Centroid Configuration 802")


env_params = EnvParams()

vehicles = VehicleParams()
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    num_vehicles=22
)

scenario = Scenario(
    name="test",
    vehicles=vehicles,
    net_params=NetParams(template="/Users/nathan/internship/I-210Pasadena/I-210subnetwork.ang")
)

env = TestEnv(env_params, sim_params, scenario, simulator='aimsun')

exp = Experiment(env)

exp.run(1, 3000)
