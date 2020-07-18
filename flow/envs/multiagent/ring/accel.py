"""Environment for training the acceleration behavior of vehicles in a ring."""
import numpy as np
from gym.spaces import Box

from flow.core import rewards
from flow.envs.ring.accel import AccelEnv
from flow.envs.multiagent.base import MultiEnv


ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 1,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 1,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 20,
}


class AdversarialAccelEnv(AccelEnv, MultiEnv):
    """Adversarial multi-agent acceleration env.

    States
        The observation of both the AV and adversary agent consist of the
        velocities and absolute position of all vehicles in the network. This
        assumes a constant number of vehicles.

    Actions
        * AV: The action space of the AV agent consists of a vector of bounded
          accelerations for each autonomous vehicle. In order to ensure safety,
          these actions are further bounded by failsafes provided by the
          simulator at every time step.
        * Adversary: The action space of the adversary agent consists of a
          vector of perturbations to the accelerations issued by the AV agent.
          These are directly added to the original accelerations by the AV
          agent.

    Rewards
        * AV: The reward for the AV agent is equal to the mean speed of all
          vehicles in the network.
        * Adversary: The adversary receives a reward equal to the negative
          reward issued to the AV agent.

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
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
            self.k.vehicle.get_speed(veh_id) / self.k.network.max_speed(),
            self.k.vehicle.get_x_by_id(veh_id) / self.k.network.length()
        ] for veh_id in self.sorted_ids])
        state = np.ndarray.flatten(state)
        return {'av': state, 'adversary': state}


class MultiAgentAccelPOEnv(MultiEnv):
    """Multi-agent acceleration environment for shared policies.

    This environment can used to train autonomous vehicles to achieve certain
    desired speeds in a decentralized fashion. This should be applicable to
    both closed and open network settings.

    Required from env_params:

    * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
    * max_decel: maximum deceleration for autonomous vehicles, in m/s^2
    * target_velocity: desired velocity for all vehicles in the network, in m/s

    States
        The observation of each agent (i.e. each autonomous vehicle) consists
        of the speeds and bumper-to-bumper headways of the vehicles immediately
        preceding and following autonomous vehicle, as well as the absolute
        position and ego speed of the autonomous vehicles. This results in a
        state space of size 6 for each agent.

    Actions
        The action space for each agent consists of a scalar bounded
        acceleration for each autonomous vehicle. In order to ensure safety,
        these actions are further bounded by failsafes provided by the
        simulator at every time step.

    Rewards
        The reward function is the two-norm of the distance of the speed of the
        vehicles in the network from the "target_velocity" term. For a
        description of the reward, see: flow.core.rewards.desired_speed. This
        reward is shared by all agents.

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        self.leader = []
        self.follower = []

        super().__init__(env_params, sim_params, network, simulator)

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-abs(self.env_params.additional_params["max_decel"]),
            high=self.env_params.additional_params["max_accel"],
            shape=(1, ),
            dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        return Box(low=-5, high=5, shape=(6,), dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        for veh_id in self.k.vehicle.get_rl_ids():
            self.k.vehicle.apply_acceleration(veh_id, rl_actions[veh_id])

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # Compute the common reward.
        reward = rewards.desired_velocity(self, fail=kwargs['fail'])

        # Reward is shared by all agents.
        return {key: reward for key in self.k.vehicle.get_rl_ids()}

    def get_state(self, **kwargs):  # FIXME
        """See class definition."""
        self.leader = []
        self.follower = []
        obs = {}

        # normalizing constants
        max_speed = self.k.network.max_speed()
        max_length = self.k.network.length()

        for rl_id in self.k.vehicle.get_rl_ids():
            this_pos = self.k.vehicle.get_x_by_id(rl_id)
            this_speed = self.k.vehicle.get_speed(rl_id)
            lead_id = self.k.vehicle.get_leader(rl_id)
            follower = self.k.vehicle.get_follower(rl_id)

            if lead_id in ["", None]:
                # in case leader is not visible
                lead_speed = max_speed
                lead_head = max_length
            else:
                self.leader.append(lead_id)
                lead_speed = self.k.vehicle.get_speed(lead_id)
                lead_head = self.k.vehicle.get_x_by_id(lead_id) \
                    - self.k.vehicle.get_x_by_id(rl_id) \
                    - self.k.vehicle.get_length(rl_id)

            if follower in ["", None]:
                # in case follower is not visible
                follow_speed = 0
                follow_head = max_length
            else:
                self.follower.append(follower)
                follow_speed = self.k.vehicle.get_speed(follower)
                follow_head = self.k.vehicle.get_headway(follower)

            # Add the next observation.
            obs[rl_id] = np.array([
                this_pos / max_length,
                this_speed / max_speed,
                (lead_speed - this_speed) / max_speed,
                lead_head / max_length,
                (this_speed - follow_speed) / max_speed,
                follow_head / max_length
            ])

        return obs

    def additional_command(self):
        """See parent class.

        This method defines which vehicles are observed for visualization
        purposes.
        """
        # specify observed vehicles
        for veh_id in self.leader + self.follower:
            self.k.vehicle.set_observed(veh_id)

    def reset(self):
        """See parent class.

        In addition, a few variables that are specific to this class are
        emptied before they are used by the new rollout.
        """
        self.leader = []
        self.follower = []
        return super().reset()
