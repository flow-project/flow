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
            loc: np.nan
            for loc in self.inflow_locations
        }
        self.outflow_locations = [
            "e_2_sbc-_0", "e_2_sbc-_1",  # east bound
            "e_4_sbc-_0", "e_4_sbc-_1",  # south bound
            "e_6_sbc-_0", "e_6_sbc-_1",  # west bound
            "e_8_sbc-_0", "e_8_sbc-_1",  # north bound
        ]
        self.outflow_values = {
            loc: np.nan
            for loc in self.outflow_locations
        }
        self.alpha = 0.5

    # ACTION GOES HERE
    @property
    def action_space(self):
        return Box(
            low=-abs(self.env_params.additional_params["max_decel"]),
            high=self.env_params.additional_params["max_accel"],
            shape=9,
            dtype=np.float32)

    def set_action(self, action):
        self.sbc_reference = {
            loc: action[idx]
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
            low=0,
            high=1,
            shape=16,
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
        return np.asarray(_inflow + _outflow)

    # REWARD FUNCTION GOES HERE
    def compute_reward(self, actions, **kwargs):
        if np.nan in list(self.inflow_values.values()) or \
           np.nan in list(self.outflow_values.values()):
            return 0
        else:
            _inflow = np.asarray([
                self.inflow_values[loc]
                for loc in self.inflow_locations
            ])
            _outflow = np.asarray([
                self.outflow_values[loc]
                for loc in self.outflow_locations
            ])
            _delay = _inflow - _outflow
            latency = -np.std(_delay)
            throughput = -np.mean(_delay)
            return self.alpha*throughput + (1 - self.alpha)*latency

    # UTILITY FUNCTION GOES HERE
    def additional_command(self):
        # update inflow statistics
        for idx, loc in enumerate(self.inflow_locations):
            _speed = self.traci_connection.lane.getLastStepMeanSpeed(loc)
            _count = self.traci_connection.lane.getLastStepVehicleNumber(loc)
            _length = self.traci_connection.lane.getLength(loc)
            _density = _count / _length
            self.inflow_values[loc] = _speed * _density

        # update outflow statistics
        for idx, loc in enumerate(self.outflow_locations):
            _speed = self.traci_connection.lane.getLastStepMeanSpeed(loc)
            _count = self.traci_connection.lane.getLastStepVehicleNumber(loc)
            _length = self.traci_connection.lane.getLength(loc)
            _density = _count / _length
            self.outflow_values[loc] = _speed * _density

        # update traffic lights state
        self.tls_state =\
            self.traci_connection.trafficlight.\
            getRedYellowGreenState(self.tls_id)

        # disable skip to test traci tls and sbc setter methods
        self.test_sbc(skip=True)
        self.test_tls(skip=True)

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

    def test_ioflow(self, skip=False):
        # TODO: test inflow outflow calculations are accurate

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
        self.get_observation(**kwargs)
