"""Contains an experiment class for running simulations."""
import datetime
import logging
import time
import os
import sys

if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


from flow.core.util import emission_to_csv
from flow.utils.registry import make_create_env


class Experiment:
    """
    Class for systematically running simulations in any supported simulator.

    This class acts as a runner for a network and environment. In order to use
    it to run an network and environment in the absence of a method specifying
    the actions of RL agents in the network, type the following:

        >>> from flow.envs import Env
        >>> flow_params = dict(...)  # see the examples in exp_config
        >>> exp = Experiment(flow_params)  # for some experiment configuration
        >>> exp.run(num_runs=1)

    If you wish to specify the actions of RL agents in the network, this may be
    done as follows:

        >>> rl_actions = lambda state: 0  # replace with something appropriate
        >>> exp.run(num_runs=1, rl_actions=rl_actions)

    Finally, if you would like to like to plot and visualize your results, this
    class can generate csv files from emission files produced by sumo. These
    files will contain the speeds, positions, edges, etc... of every vehicle
    in the network at every time step.

    In order to ensure that the simulator constructs an emission file, set the
    ``emission_path`` attribute in ``SimParams`` to some path.

        >>> from flow.core.params import SimParams
        >>> flow_params['sim'] = SimParams(emission_path="./data")

    Once you have included this in your environment, run your Experiment object
    as follows:

        >>> exp.run(num_runs=1, convert_to_csv=True)

    After the experiment is complete, look at the "./data" directory. There
    will be two files, one with the suffix .xml and another with the suffix
    .csv. The latter should be easily interpretable from any csv reader (e.g.
    Excel), and can be parsed using tools such as numpy and pandas.

    Attributes
    ----------
    env : flow.envs.Env
        the environment object the simulator will run
    custom_callables : [str, lambda]
        List of strings and lambda functions corresponding to some information we want
        to extract from the environment. The lambda will be called at each step to extract
        information from the env and it will be stored in a dict keyed by the str.
    """

    def __init__(self, flow_params, custom_callables=None):
        """Instantiate Experiment."""
        # Get the env name and a creator for the environment.
        create_env, _ = make_create_env(flow_params)
        # Take a list of
        self.custom_callables = custom_callables

        # Create the environment.
        self.env = create_env()

        logging.info(" Starting experiment {} at {}".format(
            self.env.network.name, str(datetime.datetime.utcnow())))

        logging.info("Initializing environment.")

    def run(self, num_runs, rl_actions=None, convert_to_csv=False):
        """Run the given network for a set number of runs.

        Parameters
        ----------
        num_runs : int
            number of runs the experiment should perform
        rl_actions : method, optional
            maps states to actions to be performed by the RL agents (if
            there are any)
        convert_to_csv : bool
            Specifies whether to convert the emission file created by sumo
            into a csv file

        Returns
        -------
        info_dict : dict
            contains returns, average speed per step
        """
        num_steps = self.env.env_params.horizon

        # raise an error if convert_to_csv is set to True but no emission
        # file will be generated, to avoid getting an error at the end of the
        # simulation
        if convert_to_csv and self.env.sim_params.emission_path is None:
            raise ValueError(
                'The experiment was run with convert_to_csv set '
                'to True, but no emission file will be generated. If you wish '
                'to generate an emission file, you should set the parameter '
                'emission_path in the simulation parameters (SumoParams or '
                'AimsunParams) to the path of the folder where emissions '
                'output should be generated. If you do not wish to generate '
                'emissions, set the convert_to_csv parameter to False.')

        info_dict = {}
        if rl_actions is None:
            def rl_actions(*_):
                return None

        rets = []
        mean_rets = []
        ret_lists = []
        vels = []
        mean_vels = []
        std_vels = []
        outflows = []
        custom_vals = {key: [] for key in [custom_val[0] for custom_val in self.custom_callables]}
        lambda_keys = [custom_val[0] for custom_val in self.custom_callables]
        t = time.time()
        times = []
        vehicle_times = []
        for i in range(num_runs):
            vel = np.zeros(num_steps)
            logging.info("Iter #" + str(i))
            ret = 0
            ret_list = []
            state = self.env.reset()
            for j in range(num_steps):
                t0 = time.time()
                state, reward, done, _ = self.env.step(rl_actions(state))
                t1 = time.time()
                times.append(1 / (t1 - t0))
                vehicle_times.append(self.env.k.vehicle.num_vehicles / (t1 - t0))
                vel[j] = np.mean(
                    self.env.k.vehicle.get_speed(self.env.k.vehicle.get_ids()))
                ret += reward
                ret_list.append(reward)

                for name, lambda_func in self.custom_callables:
                    val = lambda_func(self.env)
                    custom_vals[name].append(val if not np.isnan(val) else 0)

                if done:
                    break
            rets.append(ret)
            vels.append(vel)
            mean_rets.append(np.mean(ret_list))
            ret_lists.append(ret_list)
            mean_vels.append(np.mean(vel))
            std_vels.append(np.std(vel))
            outflows.append(self.env.k.vehicle.get_outflow_rate(int(500)))
            print("Round {0}, return: {1}".format(i, ret))

        info_dict["returns"] = rets
        info_dict["velocities"] = vels
        info_dict["mean_returns"] = mean_rets
        info_dict["per_step_returns"] = ret_lists
        info_dict["mean_outflows"] = np.mean(outflows)

        print("Average, std return:    {}, {}".format(
            np.mean(rets), np.std(rets)))
        print("Average, std speed:     {}, {}".format(
            np.mean(mean_vels), np.std(mean_vels)))
        for key in lambda_keys:
            info_dict[key] = custom_vals[key]
            print("Average {}, std {} for {}".format(np.mean(custom_vals[key]), np.std(custom_vals[key]), key))
            fig = plt.figure()
            plt.plot(np.arange(len(custom_vals[key])) * self.env.sim_params.sim_step, custom_vals[key])
            plt.xlabel('Time (seconds)')
            plt.ylabel(key)
            if not os.path.exists('plots'):
                os.makedirs('plots')
            plt.savefig('plots/{}.png'.format(key))
            plt.close(fig)

        print("Total time:            ", time.time() - t)
        print("steps/second:          ", np.mean(times))
        print("vehicles.steps/second: ", np.mean(vehicle_times))
        self.env.terminate()

        if convert_to_csv and self.env.simulator == "traci":
            # wait a short period of time to ensure the xml file is readable
            time.sleep(0.1)

            # collect the location of the emission file
            dir_path = self.env.sim_params.emission_path
            emission_filename = \
                "{0}-emission.xml".format(self.env.network.name)
            emission_path = os.path.join(dir_path, emission_filename)

            # convert the emission file into a csv
            emission_to_csv(emission_path)

            # Delete the .xml version of the emission file.
            os.remove(emission_path)

        return info_dict
