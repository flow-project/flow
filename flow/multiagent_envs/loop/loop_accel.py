"""Environment for training the acceleration behavior of vehicles in a loop."""

import numpy as np
from flow.core import rewards
from flow.envs.loop.loop_accel import AccelEnv
from flow.multiagent_envs.multiagent_env import MultiEnv


class MultiAgentAccelEnv(AccelEnv, MultiEnv):
    """Adversarial multi-agent env.

    Multi-agent env with an adversarial agent perturbing
    the accelerations of the autonomous vehicle
    """

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        sorted_rl_ids = [
            veh_id for veh_id in self.sorted_ids
            if veh_id in self.k.vehicle.get_rl_ids()
        ]
        av_action = rl_actions['av']
        adv_action = rl_actions['adversary']
        perturb_weight = self.env_params.additional_params['perturb_weight']
        rl_action = av_action + perturb_weight * adv_action
        self.k.vehicle.apply_acceleration(sorted_rl_ids, rl_action)

    def compute_reward(self, rl_actions, **kwargs):
        """Compute opposing rewards for agents.

        The agent receives the class definition reward,
        the adversary receives the negative of the agent reward
        """
        if self.env_params.evaluate:
            reward = np.mean(self.k.vehicle.get_speed(
                self.k.vehicle.get_ids()))
            return {'av': reward, 'adversary': -reward}
        else:
            reward = rewards.desired_velocity(self, fail=kwargs['fail'])
            return {'av': reward, 'adversary': -reward}

    def get_state(self, **kwargs):
        """See class definition for the state.

        The adversary state and the agent state are identical.
        """
        state = np.array([[
            self.k.vehicle.get_speed(veh_id) / self.k.scenario.max_speed(),
            self.k.vehicle.get_x_by_id(veh_id) / self.k.scenario.length()
        ] for veh_id in self.sorted_ids])
        state = np.ndarray.flatten(state)
        return {'av': state, 'adversary': state}
