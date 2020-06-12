from flow.networks.ring import RingNetwork
from flow.core.params import VehicleParams
from flow.controllers.car_following_models import IDMController
from flow.controllers.routing_controllers import ContinuousRouter
from flow.networks.ring import ADDITIONAL_NET_PARAMS
from flow.core.params import NetParams
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from flow.envs.ring.accel import AccelEnv
from flow.core.params import SumoParams
from flow.envs.ring.accel import ADDITIONAL_ENV_PARAMS
from flow.core.params import EnvParams
from flow.core.experiment import Experiment
import os
import pandas as pd

#define name of exp
name = "ring_example"

#create vehicles object
vehicles = VehicleParams()
vehicles.add("human",
             acceleration_controller=(IDMController, {}),
             routing_controller=(ContinuousRouter, {}),
             num_vehicles=22)

#set network parmaters
#print(ADDITIONAL_NET_PARAMS)
net_params = NetParams(additional_params=ADDITIONAL_NET_PARAMS)

#set initial condtions
initial_config = InitialConfig(spacing="uniform", perturbation=1)

#traffic light settings
traffic_lights = TrafficLightParams()

#set SUMO sim paramaters
sim_params = SumoParams(sim_step=0.1, render=True, emission_path='data')

#set environment paramters
#print(ADDITIONAL_ENV_PARAMS)
env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

#set flow experiment parameters
flow_params = dict(
    exp_tag='ring_example',
    env_name=AccelEnv,
    network=RingNetwork,
    simulator='traci',
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
    initial=initial_config,
    tls=traffic_lights,
)

# number of time steps
flow_params['env'].horizon = 3000

#create experiment obj using flow params
exp = Experiment(flow_params)

# run the sumo simulation
_ = exp.run(1, convert_to_csv=True)

# to get data file as csv
emission_location = os.path.join(exp.env.sim_params.emission_path, exp.env.network.name)
print(emission_location + '-emission.xml')
pd.read_csv(emission_location + '-emission.csv')

