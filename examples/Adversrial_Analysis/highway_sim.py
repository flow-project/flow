"""
@authors: George Gunter and Sadman Ahmed Shanto
"""
from flow.controllers import IDMController,ACC_Switched_Controller_Attacked,IDMController_Set_Congestion
from flow.core.params import EnvParams
from flow.core.params import NetParams
from flow.core.params import InitialConfig
from flow.core.params import InFlows
from flow.core.params import VehicleParams
from flow.core.params import SumoParams
from flow.core.params import SumoLaneChangeParams
from flow.core.params import SumoCarFollowingParams
from flow.networks import HighwayNetwork
from flow.envs import TestEnv
from flow.networks.highway import ADDITIONAL_NET_PARAMS
from flow.core.experiment import Experiment


# import numpy as np
# import pandas as pd
# import os, sys
# import Process_Flow_Outputs as PFO
# import time

class Simulation:
 
    def __init__(self,
        benign_params = (IDMController_Set_Congestion, {'a':1.3,'b':2.0,'noise':0.1}),
        attack_params = (ACC_Switched_Controller_Attacked, {}),
        horizon = 10000,
        sim_step =.1,
        initial_num_vehicles = 149,
        speed_limit=10.0,
        benign_inflow = 2300,
        attack_inflow = 200):

        self.benign_params = benign_params
        self.attack_params = attack_params
        self.horizon = horizon
        self.sim_step = sim_step
        self.initial_num_vehicles = initial_num_vehicles
        self.speed_limit = speed_limit
        self.benign_inflow = benign_inflow
        self.attack_inflow = self.attack_inflow
        self.runSim()

    def addVehicles(self):

        vehicles = VehicleParams()
        vehicles.add(
            veh_id="human",
            acceleration_controller=self.benign_params,
            lane_change_params=SumoLaneChangeParams(
                model="SL2015",
                lc_sublane=2.0,
            ),
        )
        vehicles.add("ACC",
        acceleration_controller=(ACC_Switched_Controller_Attacked, {
            'switch_param_time': CONGESTION_PERIOD,
            'noise': 0.3 if INCLUDE_NOISE else 0.0
        }),
        car_following_params=SumoCarFollowingParams(
            min_gap=0.1
        ),
        lane_change_params=SumoLaneChangeParams(
            model="SL2015",
            lc_sublane=2.0,
        ),
    )
        return vehicles

    def addInflows(self):
        inflow = InFlows()
        inflow.add(
            veh_type="human",
            edge="highway_0",
            vehs_per_hour=self.traffic_flow,
            departLane="free",
            departSpeed=self.traffic_speed)
        return inflow

    def setFlowParams(self, inflow,  vehicles): 
        flow_params = dict(
            # name of the experiment
            exp_tag='highway',
            # name of the flow environment the experiment is running on
            env_name=LaneChangeAccelEnv,
            # name of the network class the experiment is running on
            network=HighwayNetwork_Modified,
            # simulator that is used by the experiment
            simulator='traci',
            # sumo-related parameters (see flow.core.params.SumoParams)
            sim=SumoParams(
                sim_step=self.sim_step,
                render=False,
                lateral_resolution=0.1,
                emission_path='data/',
                restart_instance=True,
                use_ballistic=True
            ),
            # environment related parameters (see flow.core.params.EnvParams)
            env=EnvParams(
                horizon=self.sim_length,
                additional_params=ADDITIONAL_ENV_PARAMS.copy(),
            ),
            # network-related parameters (see flow.core.params.NetParams and the
            # network's documentation or ADDITIONAL_NET_PARAMS component)
            net=NetParams(
                inflows=inflow,
                additional_params= self.additional_net_params,
            ),
            # vehicles to be placed in the network at the start of a rollout (see
            # flow.core.params.VehicleParams)
            veh=vehicles,
            # parameters specifying the positioning of vehicles upon initialization/
            # reset (see flow.core.params.InitialConfig)
            initial=InitialConfig(
                spacing="uniform",
                shuffle=True,
            ),
        )
        return flow_params

    def runSim(self):
        vehicles = self.addVehicles()
        inflow = self.addInflows()
        flow_params = self.setFlowParams(inflow,vehicles)
        exp = Experiment(flow_params)
        _ = exp.run(1, convert_to_csv=True)
        emission_location = os.path.join(exp.env.sim_params.emission_path, exp.env.network.name)
        pd.read_csv(emission_location + '-emission.csv')
        self.csvFileName = emission_location+"-emission.csv"
        self.processMacroData(self.csvFileName)
        return self
