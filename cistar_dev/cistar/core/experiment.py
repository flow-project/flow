import logging
import datetime

from cistar.core.generator import Generator

"""
Acts as a runner for a scenario and environment.
"""
class SumoExperiment():

    def __init__(self, env, scenario):

        # TODO: NEEDS TO BE UPDATED
        """
        name : tag to associate experiment with
        env_class : Environment to be initialized
        num_vehicles : number of total vehicles in the simulation
        num_rl_vehicles :  these vehicles do not have a controller assigned to them
        type_params : vehicle type string -> controller (method that takes in state, returns action)
            example:
             num_rl_vehicles=4
             num_vehicles=10
             vehicle params:
               'acc' -> (2, acc_controller_fn)
               'human-0.5-delay' -> (2, human-delay)
               'human-1.0-delay' -> (2,human-delay)
               'rl' -> (4, None)

        sumo_binary : path to sumo executable
        sumo_params : parameters to pass to sumo, e.g. step-length (can also be in sumo-cfg)
        file_generator : Child of Generator that will create the net, cfg files
        cfg : specify a configuration, rather than create a new one
        """
        self.name = scenario.name
        self.num_vehicles = scenario.num_vehicles
        self.env = env
        self.type_params = scenario.type_params
        self.cfg = scenario.cfg

        logging.info(" Starting experiment" + str(self.name) + " at " + str(datetime.datetime.utcnow()))

        logging.info("initializing environment.")

    def getCfg(self):
        return self.cfg

    def getEnv(self):
        return self.env

    def run(self, num_runs, num_steps, rl_actions=[]):
        for i in range(num_runs):
            logging.info("Iter #" + str(i))
            for j in range(num_steps):
                self.env.step(rl_actions)
            self.env.reset
