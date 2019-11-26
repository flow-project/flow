"""Calibrate LWR with Gipps"""
import pandas as pd
import numpy as np
from flow.core.macroscopic import LWR
from flow.core.macroscopic.lwr import PARAMS as LWR_PARAMS
from flow.core.macroscopic.utils import run
params = LWR_PARAMS.copy()


## mean positions from csv

## mean velocities from csv

## get densities

## get velocities


##################################################################
## run lwr model one step? and get densities and velocities
## parameters to be tuned
params['length'] = 2000
params['dx'] = 0.1
params['rho_max'] = 1
params['v_max'] = 11
params['CFL'] = 0.95
params['total_time'] = 300.2
params['dt'] = 0.1
params['initial_conditions'] = #TO DO
params["boundary_conditions"] = "loop"

env = LWR(params)
obs, rew, done, _ = env.step(rl_actions = params['v_max']) #one time step
lwr_density = obs[:int(obs.shape[0]/2)]
lwr_speeds = obs[int(obs.shape[0]/2):]
#################################################################

## compare and calibrate
# at fist time step, second time step etc


## loss function

