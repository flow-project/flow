Creating Multiagent Environments
================================
In addition to single agent control, Flow supports the use of
multiple agents with individual controllers and distinct rewards.

To do this, we use RLlib's multiagent support.
The changes for multiagent environments are as follows:
instead of receiving a list of actions and returning a single observation
and a single reward, we now receive a dictionary of actions and
return a dictionary of rewards and a dictionary of observations.

The keys of the dictionary are IDs of the agent policies.

**Note that you must also subclass MultiEnv.**

A brief example of a multiagent environment:
::

    from flow.envs.multiagent.base import MultiEnv

    class MultiAgentAccelEnv(AccelEnv, MultiEnv):
        """Example multiagent environment"""

        def _apply_rl_actions(self, rl_actions):
            """
            Apply individual agent actions.

            :param rl_actions: dictionary of format {agent_id : action vector}.
            """
            rl_ids = []
            rl_action_list = []
            for rl_id, action in rl_actions.items():
                rl_ids.append(rl_id)
                rl_action_list.append(action)
            self.k.vehicle.apply_acceleration(rl_ids, rl_action_list)

        def compute_reward(self, rl_actions, **kwargs):
            """In this example, all agents receive a reward of 10"""
            reward_dict = {}
            for rl_id, action in rl_actions.items():
                reward_dict[rl_id] = 10
            return reward_dict

        def get_state(self, **kwargs):
            """Every agent observes its own speed"""
            obs_dict = {}
            for rl_id in self.k.vehicle.get_rl_ids():
                obs_dict[rl_id] = self.k.vehicle.get_speed(rl_id)
            return obs_dict


For further details look at our
`multiagent examples <https://github.com/flow-project/flow/tree/master/examples/exp_configs/rl/multiagent>`_.
