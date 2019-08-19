"""
Environment used to train a stop-and-go dissipating controller.

This is the environment that was used in:

C. Wu, A. Kreidieh, K. Parvate, E. Vinitsky, A. Bayen, "Flow: Architecture and
Benchmarking for Reinforcement Learning in Traffic Control," CoRR, vol.
abs/1710.05465, 2017. [Online]. Available: https://arxiv.org/abs/1710.05465
"""

import numpy as np
from gym.spaces.box import Box
from flow.multiagent_envs.multiagent_env import MultiEnv

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration of autonomous vehicles
    'max_accel': 1,
    # maximum deceleration of autonomous vehicles
    'max_decel': 1,
    # bounds on the ranges of ring road lengths the autonomous vehicle is
    # trained on
    'ring_length': [220, 270],
}


class MultiWaveAttenuationPOEnv(MultiEnv):
    """Multiagent shared model version of WaveAttenuationPOEnv.

    Intended to work with Lord Of The Rings Scenario.
    Note that this environment current
    only works when there is one autonomous vehicle
    on each ring.

    Required from env_params: See parent class

    States
        See parent class
    Actions
        See parent class

    Rewards
        See parent class

    Termination
        See parent class

    """

    @property
    def observation_space(self):
        """See class definition."""
        return Box(low=0, high=1, shape=(3,), dtype=np.float32)

    @property
    def action_space(self):
        """See class definition."""
        add_params = self.net_params.additional_params
        num_rings = add_params['num_rings']
        return Box(
            low=-np.abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(int(self.initial_vehicles.num_rl_vehicles / num_rings), ),
            dtype=np.float32)

    def get_state(self):
        """See class definition."""
        obs = {}
        for rl_id in self.k.vehicle.get_rl_ids():
            lead_id = self.k.vehicle.get_leader(rl_id) or rl_id

            # normalizers
            max_speed = 15.
            max_length = self.env_params.additional_params['ring_length'][1]

            observation = np.array([
                self.k.vehicle.get_speed(rl_id) / max_speed,
                (self.k.vehicle.get_speed(lead_id) -
                 self.k.vehicle.get_speed(rl_id))
                / max_speed,
                self.k.vehicle.get_headway(rl_id) / max_length
            ])
            obs.update({rl_id: observation})

        return obs

    def _apply_rl_actions(self, rl_actions):
        """Split the accelerations by ring."""
        if rl_actions:
            rl_ids = list(rl_actions.keys())
            accel = list(rl_actions.values())
            self.k.vehicle.apply_acceleration(rl_ids, accel)

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # in the warmup steps
        if rl_actions is None:
            return {}

        rew = {}
        for rl_id in rl_actions.keys():
            edge_id = rl_id.split('_')[1]
            edges = self.gen_edges(edge_id)
            vehs_on_edge = self.k.vehicle.get_ids_by_edge(edges)
            vel = np.array([
                self.k.vehicle.get_speed(veh_id)
                for veh_id in vehs_on_edge
            ])
            if any(vel < -100) or kwargs['fail']:
                return 0.

            target_vel = self.env_params.additional_params['target_velocity']
            max_cost = np.array([target_vel] * len(vehs_on_edge))
            max_cost = np.linalg.norm(max_cost)

            cost = vel - target_vel
            cost = np.linalg.norm(cost)

            rew[rl_id] = max(max_cost - cost, 0) / max_cost
        return rew

    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""
        # specify observed vehicles
        for rl_id in self.k.vehicle.get_rl_ids():
            lead_id = self.k.vehicle.get_leader(rl_id) or rl_id
            self.k.vehicle.set_observed(lead_id)

    def gen_edges(self, i):
        """Return the edges corresponding to the rl id."""
        return ['top_{}'.format(i), 'left_{}'.format(i),
                'right_{}'.format(i), 'bottom_{}'.format(i)]
