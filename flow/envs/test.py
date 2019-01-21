"""Test environment used to run simulations in the absence of autonomy."""

from flow.envs.base_env import Env
from gym.spaces.box import Box
import numpy as np


class TestEnv(Env):
    """Test environment used to run simulations in the absence of autonomy.

    Required from env_params
        None

    Optional from env_params
        reward_fn : A reward function which takes an an input the environment
        class and returns a real number.

    States
        States are an empty list.

    Actions
        No actions are provided to any RL agent.

    Rewards
        The reward is zero at every step.

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    """

    @property
    def action_space(self):
        return Box(low=0, high=0, shape=(0,), dtype=np.float32)

    @property
    def observation_space(self):
        return Box(low=0, high=0, shape=(0,), dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        return

    def compute_reward(self, rl_actions, **kwargs):
        if "reward_fn" in self.env_params.additional_params:
            return self.env_params.additional_params["reward_fn"](self)
        else:
            return 0

    def get_state(self, **kwargs):
        """See class definition."""
        return np.array([])
