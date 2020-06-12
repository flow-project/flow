"""
author: Sadman Ahmed Shanto
purpose: create a simulation of heterogenous flow on 3 lane ring road
network: 3 lane ring road
vehicles: IDM and Autonomous Vehicles of constant number
lane change: Controller needed
inflows: none for this version
traffic lights: none
sim conditions:
experiment:
graphs: space time graph, velocity time graph, FD points
"""

#Import necessary libraries and files
from flow.networks.ring import RingNetwork
from flow.core.params import VehicleParams
from flow.core.params import SumoCarFollowingParams
from flow.core.params import SumoLaneChangeParams
from flow.controllers.car_following_models import IDMController
from flow.controllers.velocity_controllers import FollowerStopper
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
from flow.controllers.lane_change_controllers import SimLaneChangeController
import numpy as np

#Define name of exp
name = "3_lane_ring_road" 

#Create vehicle object and add vehicles
vehicles = VehicleParams()
vehicles.add("HV",
             acceleration_controller=(IDMController, {"T": 1.5, "noise": 0.2}),
             routing_controller=(ContinuousRouter, {}),
             car_following_params=SumoCarFollowingParams(speed_mode="obey_safe_speed"),
             lane_change_controller=(SimLaneChangeController, {}),
             lane_change_params=SumoLaneChangeParams(lane_change_mode="strategic"),
             color="blue",
             num_vehicles=10)

vehicles.add("AV",
        acceleration_controller=(FollowerStopper,{}), 
  #      acceleration_controller=(IDMController, {"T": 0.5}),
             routing_controller=(ContinuousRouter, {}),
             car_following_params=SumoCarFollowingParams(speed_mode="obey_safe_speed", sigma=0.05, min_gap=2),
             lane_change_controller=(SimLaneChangeController, {}),
             lane_change_params=SumoLaneChangeParams(lane_change_mode="strategic", lc_speed_gain=2.0),
             color="red",
             num_vehicles=5)


#Create network paramater objects for chosen network
#   pass Additional parmaters, make sure to change them if needed
additional_net_params=ADDITIONAL_NET_PARAMS.copy()
additional_net_params['lanes']= 3
additional_net_params['length']= 400

net_params = NetParams(additional_params=additional_net_params)

#set initial condtions
initial_config = InitialConfig(spacing="random",min_gap=1,shuffle=True )

#traffic light settings
traffic_lights = TrafficLightParams()

#set SUMO sim paramaters
sim_params = SumoParams(sim_step=0.1, render=True, emission_path='data')

#set environment paramters
#print(ADDITIONAL_ENV_PARAMS)
env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

#set flow experiment parameters
flow_params = dict(
    exp_tag='3_lane_ring_trial',
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

