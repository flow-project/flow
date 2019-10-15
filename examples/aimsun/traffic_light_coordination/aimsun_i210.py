
from flow.core.params import AimsunParams, NetParams, VehicleParams, EnvParams, InitialConfig
from flow.core.experiment import Experiment
from flow.networks import Network
# from flow.envs import TestEnv
from coordinatedLightEnv import CoordinatedEnv
import os


env_params = EnvParams()
initial_config = InitialConfig()
vehicles = VehicleParams()

net_params = NetParams(
    template=os.path.expanduser("scenario_one_hour.ang")
)

sim_params = AimsunParams(
    sim_step=0.1,
    render=True,
    emission_path='data',
    replication_name="Replication (one hour)",
    centroid_config_name="Centroid Configuration 8040652"
)

network = Network(
    name="template",
    net_params=net_params,
    initial_config=initial_config,
    vehicles=vehicles
)

env = CoordinatedEnv(
    env_params,
    sim_params,
    network,
    simulator="aimsun"
)

exp = Experiment(env)
exp.run(1, 1000)
