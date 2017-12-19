import logging
import datetime


class SumoExperiment:

    def __init__(self, env, scenario):
        """
        This class acts as a runner for a scenario and environment.

        Attributes
        ----------
        env: Environment type
            the environment object the simulator will run
        scenario: Scenario type
            the scenario object the simulator will run
        """
        self.name = scenario.name
        self.num_vehicles = env.vehicles.num_vehicles
        self.env = env
        self.vehicles = scenario.vehicles
        self.cfg = scenario.cfg

        logging.info(" Starting experiment" + str(self.name) + " at "
                     + str(datetime.datetime.utcnow()))

        logging.info("initializing environment.")

    def run(self, num_runs, num_steps, rl_actions=None):
        """
        Runs the given scenario for a set number of runs and a set number of
        steps per run.

        Parameters
        ----------
        num_runs: int
            number of runs the experiment should perform
        num_steps: int
            number of steps to be performs in each run of the experiment
        rl_actions: list or numpy ndarray, optional
            actions to be performed by rl vehicles in the network (if there are
            any)
        """
        if rl_actions is None:
            rl_actions = []

        for i in range(num_runs):
            logging.info("Iter #" + str(i))
            self.env.reset()
            for j in range(num_steps):
                self.env.step(rl_actions)
        self.env.terminate()
