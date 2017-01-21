from rllab_interface import SumoEnvironment
from rllab.spaces import Box

import numpy as np

import traci

import logging
import random
class SimpleVelocityEnvironment(SumoEnvironment):

    @property
    def action_space(self):
        return Box(low=-5, high=5, shape=(self.num_cars,))

    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(self.num_cars,))

    def initialize_simulation(self):
        if not self.initial_config:
            # Get initial state of each vehicle, and store in initial config object
            for veh_id in self.ids:
                edge_id = traci.vehicle.getRoadID(veh_id)
                route_id = traci.vehicle.getRouteID(veh_id)
                lane_id = traci.vehicle.getLaneID(veh_id)
                lane_index = traci.vehicle.getLaneIndex(veh_id)
                lane_pos = traci.vehicle.getLanePosition(veh_id)
                pos = traci.vehicle.getPosition(veh_id)
                angle = traci.vehicle.getAngle(veh_id)
                speed = traci.vehicle.getSpeed(veh_id)
                self.initial_config[veh_id] = (edge_id, route_id, lane_id, lane_index, lane_pos, pos, angle, speed)
        self.reset()

    def terminate(self):
        super(SimpleVelocityEnvironment, self).terminate()
        traci.close()

    def compute_reward(self, velocity):
        return -np.linalg.norm(velocity - self.env_params["target_velocity"])

    def reset(self):
        logging.info("================= resetting  =================")
        # color = 0
        for car_id in self.ids:
            edge_id, route_id, lane_id, lane_index, lane_pos, pos, angle, speed = self.initial_config[car_id]
            x, y = pos

            logging.info("Moving car " + car_id +" from " + str(traci.vehicle.getPosition(car_id)) + " to " + str(pos))
            traci.vehicle.moveToXY(car_id, edge_id, lane_index, x, y, angle)
            traci.vehicle.slowDown(car_id, 0, 1)

            # if color == 0:
            #     traci.vehicle.setColor(car_id, (255, 255, 255, 0))
            #     color +=1
            # elif color == 1:
            #     traci.vehicle.setColor(car_id, (0, 0, 255, 0))
            #     color += 1
            # elif color == 2:
            #     traci.vehicle.setColor(car_id, (255, 0, 0, 0))
            #     color += 1
            # elif color == 4:
            #     traci.vehicle.setColor(car_id, (0, 255, 0, 0))
            #     color == 0

        traci.simulationStep()

        for index, car_id in enumerate(self.rl_ids):
            logging.info("Car " + car_id + " reset to " + str(traci.vehicle.getPosition(car_id)) + ".")
            logging.info("Car with id " + car_id + " is on route " + str(traci.vehicle.getRouteID(car_id)))
            logging.info("Car with id " + car_id + " is on edge " + str(traci.vehicle.getLaneID(car_id)))
            logging.info("Car with id " + car_id + " has valid route: " + str(traci.vehicle.isRouteValid(car_id)))
            logging.info("Car with id " + car_id + " has speed: " + str(traci.vehicle.getSpeed(car_id)))
            logging.info("Car with id " + car_id + " has pos: " + str(traci.vehicle.getPosition(car_id)))
            logging.info("Car with id " + car_id + " has route: " + str(traci.vehicle.getRoute(car_id)))
            logging.info("Car with id " + car_id + " is at indeX" + str(traci.vehicle.getRouteIndex(car_id)))


    def apply_action(self, car_id, action):
        logging.info("applying action")
        traci.vehicle.slowDown(car_id, action, 1)

    def render(self):
        pass