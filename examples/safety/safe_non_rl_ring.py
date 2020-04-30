
from flow.controllers import IDMController, ContinuousRouter
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams
from flow.envs.ring.accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.networks.ring import RingNetwork, ADDITIONAL_NET_PARAMS
from flow.controllers.car_following_models import SimCarFollowingController
from flow.core.params import SumoParams, SumoCarFollowingParams, NetParams
from flow.core.experiment import Experiment
import numpy as np

# rom the main flow directory, run;
# python flow/visualize/visualizer_rllib.py examples/safety/data/trained_ring 200 --horizon 2000


class safeEnv(AccelEnv):
    """TODO: Try to overwrite EmergencyDecel with self.k.kernel_api.vehicle.setEmergencyDecel(self, vehID, decel):"""

    def step(self, rl_actions, teleport_=True):
        """see parent class"""
        # Note: teleport_= True spawns new fresh cars if they involved in collisions and
        # keeps the simulation going:

        for _ in range(self.env_params.sims_per_step):
            self.time_counter += 1
            self.step_counter += 1

            # perform acceleration actions for controlled human-driven vehicles
            if len(self.k.vehicle.get_controlled_ids()) > 0:
                accel = []
                for veh_id in self.k.vehicle.get_controlled_ids():
                    action = self.k.vehicle.get_acc_controller(
                        veh_id).get_action(self)
                    accel.append(action)
                self.k.vehicle.apply_acceleration(
                    self.k.vehicle.get_controlled_ids(), accel)

            # perform lane change actions for controlled human-driven vehicles
            if len(self.k.vehicle.get_controlled_lc_ids()) > 0:
                direction = []
                for veh_id in self.k.vehicle.get_controlled_lc_ids():
                    target_lane = self.k.vehicle.get_lane_changing_controller(
                        veh_id).get_action(self)
                    direction.append(target_lane)
                self.k.vehicle.apply_lane_change(
                    self.k.vehicle.get_controlled_lc_ids(),
                    direction=direction)

            # perform (optionally) routing actions for all vehicles in the
            # network, including RL and SUMO-controlled vehicles
            routing_ids = []
            routing_actions = []
            for veh_id in self.k.vehicle.get_ids():
                if self.k.vehicle.get_routing_controller(veh_id) \
                        is not None:
                    routing_ids.append(veh_id)
                    route_contr = self.k.vehicle.get_routing_controller(
                        veh_id)
                    routing_actions.append(route_contr.choose_route(self))

            self.k.vehicle.choose_routes(routing_ids, routing_actions)

            self.apply_rl_actions(rl_actions)

            self.additional_command()

            # advance the simulation in the simulator by one step
            self.k.simulation.simulation_step()

            # store new observations in the vehicles and traffic lights class
            self.k.update(reset=False)

            # update the colors of vehicles
            if self.sim_params.render:
                self.k.vehicle.update_vehicle_colors()

            # crash encodes whether the simulator experienced a collision
            crash = self.k.simulation.check_collision()
            self.count_collision(teleport_)
            self.check_emergency_brakings()
            self.check_current_gaps()

            # stop collecting new simulation steps if there is a collision
            if crash and teleport_ is False:
                break

            # render a frame
            self.render()

        states = self.get_state()

        # collect information of the state of the network based on the
        # environment class used
        self.state = np.asarray(states).T

        # collect observation new state associated with action
        next_observation = np.copy(states)

        # test if the environment should terminate due to a collision or the
        # time horizon being met
        if teleport_:
            done = (self.time_counter >= self.env_params.sims_per_step *
                    (self.env_params.warmup_steps + self.env_params.horizon))
        else:
            done = (self.time_counter >= self.env_params.sims_per_step *
                    (self.env_params.warmup_steps + self.env_params.horizon)
                    or crash)

        # compute the info for each agent
        infos = {}

        # compute the reward
        if self.env_params.clip_actions:
            rl_clipped = self.clip_actions(rl_actions)
            reward = self.compute_reward(rl_clipped, fail=crash)
        else:
            reward = self.compute_reward(rl_actions, fail=crash)

        return next_observation, reward, done, infos

    def count_collision(self, teleport_):
        """count vehicle collissions"""

        if not teleport_:
            # stopping = self.k.kernel_api.simulation.getEmergencyStoppingVehiclesIDList()
            # colliding_List = self.k.kernel_api.simulation.getCollidingVehiclesIDList()
            colliding_number = self.k.kernel_api.simulation.getCollidingVehiclesNumber()
            if colliding_number > 0:
                print(str(colliding_number) + " vehicles collided")
        else:
            # colliding_List = self.k.kernel_api.simulation.getStartingTeleportIDList()
            colliding_number = self.k.kernel_api.simulation.getStartingTeleportNumber()
            if colliding_number > 0:
                print(str(colliding_number) + " vehicles collided --teleported")

        return colliding_number

    def check_emergency_brakings(self):
        """count how many cars performed emergency braking"""

        current_accel = []
        max_decel = []
        for veh_id in self.k.vehicle.get_ids():
            max_decel = np.append(max_decel, self.k.kernel_api.vehicle.getEmergencyDecel(veh_id))
            current_accel = np.append(current_accel,  self.k.kernel_api.vehicle.getAcceleration(veh_id))

        round_off_accel = np.round(current_accel)
        # print(round_off_accel, current_accel) #FIXME
        fast_brakings = np.sum(round_off_accel*-1 == max_decel)

        if fast_brakings > 0:
            print(str(fast_brakings) + " emergency brakings")

        return fast_brakings

    def check_current_gaps(self):

        gaps = []
        for veh_id in self.k.vehicle.get_ids():
            # env.k.vehicle.get_speed(self.veh_id)
            gaps = np.append(gaps, self.k.vehicle.get_headway(veh_id))
        # print(min(gaps))
        return gaps


"""Used as an example of ring experiment.

This example consists of 22 IDM cars on a ring creating shockwaves.
"""

vehicles = VehicleParams()
# vehicles.add(
#     veh_id="idm",
#     acceleration_controller=(IDMController, {}),
#     routing_controller=(ContinuousRouter, {}),
#     num_vehicles=21)

# vehicles.add(
#     veh_id="idm2",
#     acceleration_controller=(IDMController, {"T": 0.0001, "s0": -1}),
#     routing_controller=(ContinuousRouter, {}),
#     num_vehicles=1)


vehicles.add(
    veh_id="idm",
    acceleration_controller=(SimCarFollowingController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        carFollowModel="IDM",
    ),
    num_vehicles=20)


# RLCar needs to be inserted here and somehow passed into params.json

vehicles.add(
    veh_id="idm2",
    acceleration_controller=(SimCarFollowingController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        carFollowModel="IDM", minGap=0, tau=0.001, accel=20, max_speed=300, decel=1.5,
    ),
    num_vehicles=1)

# carFollowModel="IDM", minGap=0, tau=0.1, accel=20, max_speed=300, decel=1.5, # causes acrash
# carFollowModel="IDM", minGap=0, tau=1, accel=20, max_speed=300, decel=1.5, # causes max decel

flow_params = dict(
    # name of the experiment
    exp_tag='ring',

    # name of the flow environment the experiment is running on
    env_name=safeEnv,

    # name of the network class the experiment is running on
    network=RingNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        render=True,
        sim_step=0.1,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=3000,
        additional_params=ADDITIONAL_ENV_PARAMS,
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        additional_params=ADDITIONAL_NET_PARAMS.copy(),
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        bunching=20,
    ),
)

# number of time steps
exp = Experiment(flow_params)

# run the sumo simulation
_ = exp.run(1, convert_to_csv=False)
