"""Two test environments designed to fail abstract base class tests."""

from flow.envs.base import Env

import numpy as np


class TestFailRLActionsEnv(Env):
    """Test environment designed to fail _apply_rl_actions not-implemented test."""

    @property
    def action_space(self):
        """See parent class."""
        return Box(low=0, high=0, shape=(0,), dtype=np.float32)

    @property
    def observation_space(self):
        """See parent class."""
        return Box(low=0, high=0, shape=(0,), dtype=np.float32)

    def get_state(self, **kwargs):
        """See class definition."""
        return np.array([])


class TestFailGetStateEnv(Env):
    """Test environment designed to fail get_state not-implemented test."""

    @property
    def action_space(self):
        """See parent class."""
        return Box(low=0, high=0, shape=(0,), dtype=np.float32)

    @property
    def observation_space(self):
        """See parent class."""
        return Box(low=0, high=0, shape=(0,), dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        return
