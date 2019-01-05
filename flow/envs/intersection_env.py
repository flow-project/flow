"""Environment for training the acceleration behavior of vehicles in a loop."""

from flow.envs.base_env import Env
from flow.core import rewards
from flow.core.params import InitialConfig, NetParams, SumoCarFollowingParams
from flow.controllers import IDMController

from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple

import numpy as np

import os
from os.path import expanduser
HOME = expanduser("~")
import time

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 5,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 11.176,
}


class IntersectionEnv(Env):
    def __init__(self, env_params, sumo_params, scenario):
        print("Starting IntersectionEnv...")
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        super().__init__(env_params, sumo_params, scenario)

        # setup traffic lights
        self.tls_id = self.traci_connection.trafficlight.getIDList()[0]
        self.tls_state =\
            self.traci_connection.trafficlight.\
            getRedYellowGreenState(self.tls_id)
        self.tls_definition =\
            self.traci_connection.trafficlight.\
            getCompleteRedYellowGreenDefinition(self.tls_id)
        self.tls_phase = 0
        self.tls_phase_count = 0
        for logic in self.tls_definition:
            for phase in logic._phases:
                self.tls_phase_count += 1

        # setup speed broadcasters
        self.sbc_locations = [
            "e_1_sbc+_0", "e_1_sbc+_1",  # east bound
            "e_3_sbc+_0", "e_3_sbc+_1",  # south bound
            "e_5_sbc+_0", "e_5_sbc+_1",  # west bound
            "e_7_sbc+_0", "e_7_sbc+_1",  # north bound
        ]
        # default speed reference to 11.176 m/s
        self.sbc_reference = {
            loc: self.traci_connection.lane.getMaxSpeed(loc)
            for loc in self.sbc_locations
        }

        # setup inflow outflow logger
        self.inflow_locations = [
            "e_1_sbc+_0", "e_1_sbc+_1",  # east bound
            "e_3_sbc+_0", "e_3_sbc+_1",  # south bound
            "e_5_sbc+_0", "e_5_sbc+_1",  # west bound
            "e_7_sbc+_0", "e_7_sbc+_1",  # north bound
        ]
        self.inflow_values = {
            loc: 0
            for loc in self.inflow_locations
        }
        self.outflow_locations = [
            "e_2_sbc-_0", "e_2_sbc-_1",  # east bound
            "e_4_sbc-_0", "e_4_sbc-_1",  # south bound
            "e_6_sbc-_0", "e_6_sbc-_1",  # west bound
            "e_8_sbc-_0", "e_8_sbc-_1",  # north bound
        ]
        self.outflow_values = {
            loc: 0
            for loc in self.outflow_locations
        }
        self.alpha = 0.8
        self.rewards = 0

    # ACTION GOES HERE
    @property
    def action_space(self):
        return Box(
            low=0,
            high=max(self.scenario.max_speed, self.tls_phase_count-1),
            shape=(9,),
            dtype=np.float32)

    def set_action(self, action):
        self.sbc_reference = {
            loc: np.clip(action[idx], 0, np.inf)
            for idx, loc in enumerate(self.sbc_locations)
        }
        self._set_reference(self.sbc_reference)
        self.tls_phase = np.clip(int(action[-1]), 0, self.tls_phase_count-1)
        self._set_phase(self.tls_phase)

    # OBSERVATION GOES HERE
    @property
    def observation_space(self):
        """See class definition."""
        return Box(
            low=0.,
            high=np.inf,
            shape=(16,),
            dtype=np.float32)

    def get_observation(self, **kwargs):
        _inflow = [
            self.inflow_values[loc]
            for loc in self.inflow_locations
        ]
        _outflow = [
            self.outflow_values[loc]
            for loc in self.outflow_locations
        ]
        observation = np.asarray(_inflow + _outflow)
        return observation

    # REWARD FUNCTION GOES HERE
    def get_reward(self, **kwargs):
        return -np.power(self.vehicles.num_vehicles, 0.9)

    def get_reward_deprecated(self, **kwargs):
        _inflow = np.asarray([
            self.inflow_values[loc]
            for loc in self.inflow_locations
        ])
        _outflow = np.asarray([
            self.outflow_values[loc]
            for loc in self.outflow_locations
        ])
        input_efficiency = \
            self.alpha*np.mean(_inflow) + (1 - self.alpha)*(-np.std(_inflow))
        output_efficiency = \
            self.alpha*np.mean(_outflow) + (1 - self.alpha)*(-np.std(_outflow))
        return input_efficiency + output_efficiency

    # UTILITY FUNCTION GOES HERE
    def additional_command(self):
        # update inflow statistics
        _inflow_stats = []
        for idx, loc in enumerate(self.inflow_locations):
            _speed = self.traci_connection.lane.getLastStepMeanSpeed(loc)
            _count = self.traci_connection.lane.getLastStepVehicleNumber(loc)
            _length = self.traci_connection.lane.getLength(loc)
            _density = _count / _length
            _inflow_stats.append([_speed, _count, _length, _density])
            self.inflow_values[loc] = _speed * _density

        # update outflow statistics
        _outflow_stats = []
        for idx, loc in enumerate(self.outflow_locations):
            _speed = self.traci_connection.lane.getLastStepMeanSpeed(loc)
            _count = self.traci_connection.lane.getLastStepVehicleNumber(loc)
            _length = self.traci_connection.lane.getLength(loc)
            _density = _count / _length
            _outflow_stats.append([_speed, _count, _length, _density])
            self.outflow_values[loc] = _speed * _density

        # update traffic lights state
        self.tls_state =\
            self.traci_connection.trafficlight.\
            getRedYellowGreenState(self.tls_id)

        # disable skip to test traci tls and sbc setter methods
        self.test_sbc(skip=True)
        self.test_tls(skip=True)
        self.test_ioflow(_inflow_stats, _outflow_stats, skip=True)
        self.test_reward(skip=True)

    def test_sbc(self, skip=True):
        if self.time_counter > 50 and not skip:
            print("Broadcasting reference...")
            self.sbc_reference = {
                loc: 1
                for loc in self.sbc_locations
            }
            self._set_reference(self.sbc_reference)

    def test_tls(self, skip=True):
        if self.time_counter % 10 == 0 and not skip:
            print("Switching phase...")
            self.tls_phase = np.random.randint(0, self.tls_phase_count-1)
            print("New phase:", self.tls_phase)
            self._set_phase(self.tls_phase)

    def test_ioflow(self, inflow_stats, outflow_stats, skip=False):
        if not skip:
            print(inflow_stats)
            print(self.inflow_values)
            print(outflow_stats)
            print(self.outflow_values)

    def test_reward(self, skip=True):
        if not skip:
            _reward = self.get_reward()
            print('Reward this step:', _reward)
            self.rewards += _reward
            print('Total rewards:', self.rewards)

    def _set_reference(self, sbc_reference):
        for sbc, reference in sbc_reference.items():
            sbc_clients = self.traci_connection.lane.getLastStepVehicleIDs(sbc)
            for veh_id in sbc_clients:
                self.traci_connection.vehicle.slowDown(veh_id, reference, 10)

    def _set_phase(self, tls_phase):
        self.traci_connection.trafficlight.setPhase(\
            self.tls_id, tls_phase)

    # DO NOT WORRY ABOUT ANYTHING BELOW THIS LINE >◡<
    def _apply_rl_actions(self, rl_actions):
        self.set_action(rl_actions)

    def get_state(self, **kwargs):
        return self.get_observation(**kwargs)

    def compute_reward(self, actions, **kwargs):
        return self.get_reward(**kwargs)
