"""Energy bracketing runner script for computing min/max energy savings bounds."""
from flow.networks import HighwayNetwork
from flow.core.params import NetParams, InitialConfig
from flow.networks.highway import ADDITIONAL_NET_PARAMS
from flow.envs import SingleStraightRoadEnergyBracketing
from flow.envs.straightroad_env import ADDITIONAL_ENV_PARAMS
from flow.core.params import VehicleParams
from flow.controllers import IDMController
from flow.controllers import TrajectoryFollower
from flow.core.params import SumoParams
from flow.core.params import EnvParams, SumoCarFollowingParams
from flow.data_pipeline.data_pipeline import write_dict_to_csv, get_extra_info
import argparse
from collections import defaultdict
import uuid
import numpy as np
import os
from flow.utils.registry import make_create_env
import sys

ROAD_LENGTH = 5000
INITIAL_SPEED = 12.0
N_HUMANS = 9


def parse_args(args):
    """Parse experiment options user can specify in command line.

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser(
        description="Parse argument used when running a min/max simulation.",
        epilog="python min_max_exp.py --start_amp FLOAT --end_amp FLOAT --freq FLOAT --sweep_step FLOAT --emodel STR")

    parser.add_argument(
        '--start_amp', type=float, default=0,
        help='Starting amplitude to sweep over. Defaults to 0.')
    parser.add_argument(
        '--end_amp', type=float, default=15,
        help='Ending amplitude to sweep over. Defaults to 15.')
    parser.add_argument(
        '--sweep_step', type=float, default=1,
        help='Step between amplitudes. Defaults to 1.')
    parser.add_argument(
        '--freq', type=float, default=0.1,
        help='Frequency of the sine wave. Defaults to 0.2.')
    parser.add_argument(
        '--emodel', type=str,
        help='Energy model to be used. choose from: PriusEnergy, TacomaEnergy, PDMCombustionEngine, PDMElectric')

    return parser.parse_known_args(args)[0]


if __name__ == "__main__":
    # Get Arguments from parser
    args = parse_args(sys.argv[1:])
    # assign arguments to variables
    if args.start_amp:
        s_amp = args.start_amp
        e_amp = args.end_amp
    else:
        s_amp = args.start_amp
        e_amp = args.end_amp
    if args.sweep_step:
        sweep_step = args.sweep_step
    else:
        sweep_step = 1
    if args.freq:
        freq = args.freq
    else:
        freq = 0.1
    if args.emodel:
        if args.emodel == 'PriusEnergy':
            from flow.energy_models.toyota_energy import PriusEnergy

            ENERGY_MODEL = PriusEnergy
        elif args.emodel == 'TacomaEnergy':
            from flow.energy_models.toyota_energy import TacomaEnergy

            ENERGY_MODEL = TacomaEnergy
        elif args.emodel == 'PDMElectric':
            from flow.energy_models.power_demand import PDMElectric

            ENERGY_MODEL = PDMElectric
        elif args.emodel == 'PDMCombustionEngine':
            from flow.energy_models.power_demand import PDMCombustionEngine

            ENERGY_MODEL = PDMCombustionEngine
        else:
            print('Energy model undefined. PDMCombustionEngine assigned by default')
            from flow.energy_models.power_demand import PDMCombustionEngine

            ENERGY_MODEL = PDMCombustionEngine

    network_name = HighwayNetwork
    # name of the network
    name = "bracketing_example"

    # network-specific parameters
    net_params = NetParams(additional_params=ADDITIONAL_NET_PARAMS)
    net_params.additional_params['length'] = ROAD_LENGTH
    net_params.additional_params['ghost_speed_limit'] = 30.0
    # initial configuration to vehicles
    initial_config = InitialConfig()
    # Place vehicle at the start of the lane with 15 m in between
    initial_config.spacing = "custom"
    initial_pos = {"start_positions": [],
                   "start_lanes": []}
    for i in range(N_HUMANS + 1):
        initial_pos["start_positions"].append(('highway_0', i * 20))
        initial_pos["start_lanes"].append(0)

    initial_config.additional_params = initial_pos

    # Add vehicles
    vehicles = VehicleParams()
    vehicles.add("human",
                 acceleration_controller=(IDMController, {}),
                 car_following_params=SumoCarFollowingParams(accel=4.0),
                 energy_model=ENERGY_MODEL,
                 initial_speed=INITIAL_SPEED,
                 num_vehicles=N_HUMANS)
    # accel is the maximum acceleration allowed, decel default value is 4.5
    cruse_speed = INITIAL_SPEED
    vehicles.add(veh_id="AV",
                 acceleration_controller=(TrajectoryFollower, {"func": lambda _: None}),
                 car_following_params=SumoCarFollowingParams(accel=4.0),
                 #              energy_model=ENERGY_MODEL,
                 initial_speed=INITIAL_SPEED,
                 num_vehicles=1)
    # get simulation parameters
    sim_params = SumoParams(sim_step=0.1, render=False)
    setattr(sim_params, 'num_clients', 1)
    sim_params.restart_instance = True
    # Set time horizon of the experiment
    # (make sure it's large enough for all veh to exit)
    HORIZON = 100000
    # set environment parameters
    additional_env_params = ADDITIONAL_ENV_PARAMS
    additional_env_params['max_accel'] = 4.0
    additional_env_params['max_decel'] = 4.0
    env_params = EnvParams(
        horizon=HORIZON,
        additional_params=additional_env_params
    )
    # choose environment (check definition from imports)
    env_name = SingleStraightRoadEnergyBracketing
    # log experiment parameters
    flow_params = dict(
        # name of the experiment
        exp_tag=name,
        # name of the flow environment the experiment is running on
        env_name=env_name,
        # name of the network class the experiment uses
        network=network_name,
        # simulator that is used by the experiment
        simulator='traci',
        # simulation-related parameters
        sim=sim_params,
        # environment related parameters (see flow.core.params.EnvParams)
        env=env_params,
        # network-related parameters (see flow.core.params.NetParams and
        # the network's documentation or ADDITIONAL_NET_PARAMS component)
        net=net_params,
        # vehicles to be placed in the network at the start of a rollout
        # (see flow.core.vehicles.Vehicles)
        veh=vehicles,
        # (optional) parameters affecting the positioning of vehicles upon
        # initialization/reset (see flow.core.params.InitialConfig)
        initial=initial_config
    )
    # create the environment
    create_env, gym_name = make_create_env(params=flow_params, version=0)
    env = create_env(gym_name)

    env_params.restart_instance = False
    # initiate data logging
    extra_info = defaultdict(lambda: [])
    source_id = 'flow_{}'.format(uuid.uuid4().hex)

    rets = []  # returns = [total fuel at each run]
    ftimes = []  # final times = [total time for each run]
    amplitudes = list(np.arange(s_amp, e_amp + sweep_step, sweep_step))  # amplitudes used for sweeping
    print("starting experiment with energy model {}".format(ENERGY_MODEL))
    for amplit in amplitudes:
        run_id = "run_amp_{}".format(amplit)
        env.pipeline_params = (extra_info, source_id, run_id)  # update run_id
        state = env.reset()  # reset env at each run
        ret = 0
        rl_speeds = []  # speeds of controlled vehicle as outputted from simulator
        d_speeds = []  # desired speeds
        # update the speed function for each run
        env.k.vehicle.get_acc_controller('AV_0').speed_func = \
            lambda x: cruse_speed + amplit * np.sin(freq * (env.time_counter + 1) * env.sim_step)
        # Simulate Run
        for _ in range(env_params.horizon):

            step_vehicles = env.unwrapped.k.vehicle
            state, reward, done, _ = env.step(None)

            # collect data for data pipeline
            get_extra_info(step_vehicles, extra_info, step_vehicles.get_ids(), source_id, run_id)

            ret += reward
            if done:
                break
        # log total fuel and time after run is done
        rets.append(ret)
        ftime = env.time_counter * env.sim_step
        ftimes.append(ftime)
        print('Amplitide {}, Fuel consumed: {}, Time taken: {}'.format(amplit, ret, ftime))

    # terminate env after sweeping iver all amplitudes
    env.unwrapped.terminate()
    # dump to csv
    dir_path = os.path.dirname(os.path.realpath('__file__'))
    trajectory_table_path = os.path.join(dir_path, '{}_trajectories.csv'.format(source_id))
    write_dict_to_csv(trajectory_table_path, extra_info, True)
    trajectory_table_path2 = trajectory_table_path[:-4] + '_amp_vs_fuel.csv'
    fuel_data = defaultdict(lambda: [])
    fuel_data['Ampiltude'] = amplitudes
    fuel_data['Fuel'] = rets
    fuel_data['Time'] = ftimes
    write_dict_to_csv(trajectory_table_path2, fuel_data, True)
