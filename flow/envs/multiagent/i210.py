"""Environment for training vehicles to reduce congestion in the I210."""

from gym.spaces import Box
import numpy as np

from flow.envs.multiagent.base import MultiEnv

# largest number of lanes on any given edge in the network
MAX_LANES = 6
SPEED_SCALE = 50
HEADWAY_SCALE = 1000

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 1,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 1,
    # whether we use an obs space that contains adjacent lane info or just the lead obs
    "lead_obs": True,
    # whether the reward should come from local vehicles instead of global rewards
    "local_reward": True,
    "target_velocity": 25
}


class I210MultiEnv(MultiEnv):
    """Partially observable multi-agent environment for the I-210 subnetworks.

    The policy is shared among the agents, so there can be a non-constant
    number of RL vehicles throughout the simulation.

    Required from env_params:

    * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
    * max_decel: maximum deceleration for autonomous vehicles, in m/s^2

    The following states, actions and rewards are considered for one autonomous
    vehicle only, as they will be computed in the same way for each of them.

    States
        The observation consists of the speeds and bumper-to-bumper headways of
        the vehicles immediately preceding and following autonomous vehicles in
        all of the preceding lanes as well, a binary value indicating which of
        these vehicles is autonomous, and the speed of the autonomous vehicle.
        Missing vehicles are padded with zeros.

    Actions
        The action consists of an acceleration, bound according to the
        environment parameters, as well as three values that will be converted
        into probabilities via softmax to decide of a lane change (left, none
        or right). NOTE: lane changing is currently not enabled. It's a TODO.

    Rewards
        The reward function encourages proximity of the system-level velocity
        to a desired velocity specified in the environment parameters, while
        slightly penalizing small time headways among autonomous vehicles.

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)
        self.lead_obs = env_params.additional_params.get("lead_obs")
        self.reroute_on_exit = env_params.additional_params.get("reroute_on_exit")
        self.max_lanes = MAX_LANES
        self.num_enter_lanes = 5
        self.entrance_edge = "119257914"
        self.exit_edge = "119257908#3"
        self.leader = []

    @property
    def observation_space(self):
        """See class definition."""
        # speed, speed of leader, headway
        if self.lead_obs:
            return Box(
                low=-float('inf'),
                high=float('inf'),
                shape=(3,),
                dtype=np.float32
            )
        # speed, dist to ego vehicle, binary value which is 1 if the vehicle is
        # an AV
        else:
            leading_obs = 3 * self.max_lanes
            follow_obs = 3 * self.max_lanes

            # speed and lane
            self_obs = 2

            return Box(
                low=-float('inf'),
                high=float('inf'),
                shape=(leading_obs + follow_obs + self_obs,),
                dtype=np.float32
            )

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-np.abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(1,),  # (4,),
            dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        # in the warmup steps, rl_actions is None
        if rl_actions:
            for rl_id, actions in rl_actions.items():
                accel = actions[0]

                # lane_change_softmax = np.exp(actions[1:4])
                # lane_change_softmax /= np.sum(lane_change_softmax)
                # lane_change_action = np.random.choice([-1, 0, 1],
                #                                       p=lane_change_softmax)

                self.k.vehicle.apply_acceleration(rl_id, accel)
                # self.k.vehicle.apply_lane_change(rl_id, lane_change_action)

    def get_state(self):
        """See class definition."""
        if self.lead_obs:
            veh_info = {}
            for rl_id in self.k.vehicle.get_rl_ids():
                speed = self.k.vehicle.get_speed(rl_id)
                lead_id = self.k.vehicle.get_leader(rl_id)
                if lead_id in ["", None]:
                    # in case leader is not visible
                    lead_speed = SPEED_SCALE
                    headway = HEADWAY_SCALE
                else:
                    lead_speed = self.k.vehicle.get_speed(lead_id)
                    headway = self.k.vehicle.get_headway(rl_id)
                veh_info.update({rl_id: np.array([speed / SPEED_SCALE, headway / HEADWAY_SCALE,
                                                  lead_speed / SPEED_SCALE])})
        else:
            veh_info = {rl_id: np.concatenate((self.state_util(rl_id),
                                               self.veh_statistics(rl_id)))
                        for rl_id in self.k.vehicle.get_rl_ids()}
        return veh_info

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # in the warmup steps
        if rl_actions is None:
            return {}

        rewards = {}
        if self.env_params.additional_params["local_reward"]:
            des_speed = self.env_params.additional_params["target_velocity"]
            for rl_id in self.k.vehicle.get_rl_ids():
                rewards[rl_id] = 0
                speeds = []
                follow_speed = self.k.vehicle.get_speed(self.k.vehicle.get_follower(rl_id))
                if follow_speed >= 0:
                    speeds.append(follow_speed)
                if self.k.vehicle.get_speed(rl_id) >= 0:
                    speeds.append(self.k.vehicle.get_speed(rl_id))
                if len(speeds) > 0:
                    # rescale so the critic can estimate it quickly
                    rewards[rl_id] = np.mean([(des_speed - np.abs(speed - des_speed)) ** 2
                                              for speed in speeds]) / (des_speed ** 2)
        else:
            speeds = self.k.vehicle.get_speed(self.k.vehicle.get_ids())
            des_speed = self.env_params.additional_params["target_velocity"]
            # rescale so the critic can estimate it quickly
            reward = np.nan_to_num(np.mean([(des_speed - np.abs(speed - des_speed)) ** 2
                                            for speed in speeds]) / (des_speed ** 2))
            rewards = {rl_id: reward for rl_id in self.k.vehicle.get_rl_ids()}
        return rewards

    def additional_command(self):
        """See parent class.

        Define which vehicles are observed for visualization purposes. Additionally, optionally reroute vehicles
        back once they have exited.
        """
        super().additional_command()
        # specify observed vehicles
        for rl_id in self.k.vehicle.get_rl_ids():
            # leader
            lead_id = self.k.vehicle.get_leader(rl_id)
            if lead_id:
                self.k.vehicle.set_observed(lead_id)

        if self.reroute_on_exit and self.time_counter >= self.env_params.sims_per_step * self.env_params.warmup_steps \
                and not self.env_params.evaluate:
            veh_ids = self.k.vehicle.get_ids()
            edges = self.k.vehicle.get_edge(veh_ids)
            for veh_id, edge in zip(veh_ids, edges):
                if edge == "":
                    continue
                if edge[0] == ":":  # center edge
                    continue
                # on the exit edge, near the end, and is the vehicle furthest along
                if edge == self.exit_edge and \
                        (self.k.vehicle.get_position(veh_id) > self.k.network.edge_length(self.exit_edge) - 100) \
                        and self.k.vehicle.get_leader(veh_id) is None:
                    type_id = self.k.vehicle.get_type(veh_id)
                    # remove the vehicle
                    self.k.vehicle.remove(veh_id)
                    lane = np.random.randint(low=0, high=self.num_enter_lanes)
                    # reintroduce it at the start of the network
                    # TODO(@evinitsky) select the lane and speed a bit more cleanly
                    # Note, the position is 10 so you are not overlapping with the inflow car that is being removed.
                    # this allows the vehicle to be immediately inserted.
                    self.k.vehicle.add(
                        veh_id=veh_id,
                        edge=self.entrance_edge,
                        type_id=str(type_id),
                        lane=str(lane),
                        pos="10.0",
                        speed="23.0")

            departed_ids = self.k.vehicle.get_departed_ids()
            if len(departed_ids) > 0:
                for veh_id in departed_ids:
                    if veh_id not in self.observed_ids:
                        self.k.vehicle.remove(veh_id)

    def state_util(self, rl_id):
        """Return an array of headway, tailway, leader speed, follower speed.

        Also return a 1 if leader is rl 0 otherwise, a 1 if follower is rl 0 otherwise.
        If there are fewer than MAX_LANES the extra
        entries are filled with -1 to disambiguate from zeros.
        """
        veh = self.k.vehicle
        lane_headways = veh.get_lane_headways(rl_id).copy()
        lane_tailways = veh.get_lane_tailways(rl_id).copy()
        lane_leader_speed = veh.get_lane_leaders_speed(rl_id).copy()
        lane_follower_speed = veh.get_lane_followers_speed(rl_id).copy()
        leader_ids = veh.get_lane_leaders(rl_id).copy()
        follower_ids = veh.get_lane_followers(rl_id).copy()
        rl_ids = self.k.vehicle.get_rl_ids()
        is_leader_rl = [1 if l_id in rl_ids else 0 for l_id in leader_ids]
        is_follow_rl = [1 if f_id in rl_ids else 0 for f_id in follower_ids]
        diff = MAX_LANES - len(is_leader_rl)
        if diff > 0:
            # the minus 1 disambiguates missing cars from missing lanes
            lane_headways += diff * [-1]
            lane_tailways += diff * [-1]
            lane_leader_speed += diff * [-1]
            lane_follower_speed += diff * [-1]
            is_leader_rl += diff * [-1]
            is_follow_rl += diff * [-1]
        lane_headways = np.asarray(lane_headways) / 1000
        lane_tailways = np.asarray(lane_tailways) / 1000
        lane_leader_speed = np.asarray(lane_leader_speed) / 100
        lane_follower_speed = np.asarray(lane_follower_speed) / 100
        return np.concatenate((lane_headways, lane_tailways, lane_leader_speed,
                               lane_follower_speed, is_leader_rl,
                               is_follow_rl))

    def veh_statistics(self, rl_id):
        """Return speed, edge information, and x, y about the vehicle itself."""
        speed = self.k.vehicle.get_speed(rl_id) / 100.0
        lane = (self.k.vehicle.get_lane(rl_id) + 1) / 10.0
        return np.array([speed, lane])

    def step(self, rl_actions):
        """See parent class for more details; add option to reroute vehicles."""
        state, reward, done, info = super().step(rl_actions)
        # handle the edge case where a vehicle hasn't been put back when the rollout terminates
        if self.reroute_on_exit and done['__all__']:
            for rl_id in self.observed_rl_ids:
                if rl_id not in state.keys():
                    done[rl_id] = True
                    reward[rl_id] = 0
                    state[rl_id] = -1 * np.ones(self.observation_space.shape[0])
        return state, reward, done, info


class MultiStraightRoad(I210MultiEnv):
    """Partially observable multi-agent environment for a straight road. Look at superclass for more information."""

    def __init__(self, env_params, sim_params, network, simulator):
        super().__init__(env_params, sim_params, network, simulator)
        self.num_enter_lanes = 1
        self.entrance_edge = self.network.routes['highway_0'][0][0][0]
        self.exit_edge = self.network.routes['highway_0'][0][0][-1]

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        # in the warmup steps, rl_actions is None
        if rl_actions:
            rl_ids = []
            accels = []
            for rl_id, actions in rl_actions.items():
                accels.append(actions[0])
                rl_ids.append(rl_id)

            # prevent the AV from blocking the entrance
            self.k.vehicle.apply_acceleration(rl_ids, accels)
