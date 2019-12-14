# import the base environment class
from flow.envs import Env
from gym.spaces.box import Box
import numpy as np

class SimpleEnv(Env):
    
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super(SimpleEnv, self).__init__(env_params, sim_params, network, simulator='traci')
        self.tls_ids = ['GS_27154068', "27153986", "27403668", "27154051"]
        self.phase_dict = dict()
        self.phase_dict[self.tls_ids[0]] = ["GgrrGG", "ygrryy", "rGrrrr", "ryrrrr", "rrGGGr", "rryyyr"]
        self.phase_dict[self.tls_ids[1]] = ["GGgrrrGGgrrr", "yygrrryygrrr", "rrGrrrrrGrrr", "rryrrrrryrrr", 
                                "rrrGGgrrrGGg", "rrryygrrryyg", "rrrrrGrrrrrG", "rrrrryrrrrry"]
        self.phase_dict[self.tls_ids[2]] = ["rrGGGg", "rryyyg", "rrrrrG", "rrrrry", "GGGrrr", "yyyrrr"]
        self.phase_dict[self.tls_ids[3]] = ["rrrGGgrrrGGg", "rrryyyrrryGG", "rrrrrrrrrrGG", "rrrrrrrrrryy",
                                "GGgrrrGGgrrr", "GGGrrryyyrrr", "GGGrrrrrrrrr", "yyyrrrrrrrrr"]
        self.det_ids = []
        self.veh_seen = []
        
        self.active_phase = dict()
        for tl_id in self.tls_ids:
            self.active_phase[tl_id] = 0
    
    @property
    def action_space(self):
        # return binary value. 0: no change, 1: change TL state
        # 4 TLS shall be optimized
        return Box(low=0, high=1, shape=(4,))
    
    @property
    def observation_space(self):
        number_of_ILs = len(self.k.lane_area_detector.get_ids())
        # nVehSeen reward shall be -nVehSeen, hence 1 variable per loop:
        number_of_signals = 1
        return Box(low=0, high=10000, shape=(number_of_signals * number_of_ILs,))
    
    def get_state(self):
        self.det_ids = self.k.lane_area_detector.get_ids()
        self.veh_seen = []
        for detector_id in self.det_ids:
            self.veh_seen.append(self.k.lane_area_detector.get_n_veh_seen(detector_id))
        return np.asarray(self.veh_seen)
    
    def _apply_rl_actions(self, rl_actions):
        
        for i, tl_id in enumerate(self.tls_ids):
            
            if rl_actions[i] > 0.5:
                if (self.active_phase[tl_id] < len(self.phase_dict[tl_id]) - 1):
                    self.active_phase[tl_id] += 1
                else:
                    self.active_phase[tl_id] = 0
                    
            active_phase = self.phase_dict[tl_id][self.active_phase[tl_id]]
            self.k.traffic_light.set_state(tl_id, active_phase)
        
        
    def compute_reward(self, rl_actions, **kwargs):
        reward = -np.sum(self.veh_seen)
        return reward