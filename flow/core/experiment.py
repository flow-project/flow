"""Contains an experiment class for running simulations."""
from flow.utils.registry import make_create_env
from flow.data_pipeline.data_pipeline import write_dict_to_csv, upload_to_s3, get_extra_info, get_configuration
from flow.data_pipeline.leaderboard_utils import network_name_translate
from collections import defaultdict
from datetime import datetime, timezone
import logging
import time
import os
import numpy as np
import uuid


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
    custom_callables : dict < str, lambda >
        strings and lambda functions corresponding to some information we want
        to extract from the environment. The lambda will be called at each step
        to extract information from the env and it will be stored in a dict
        keyed by the str.
    env : flow.envs.Env
        the environment object the simulator will run
    """

    def __init__(self, flow_params, custom_callables=None, register_with_ray=False):
        """Instantiate the Experiment class.

        Parameters
        ----------
        flow_params : dict
            flow-specific parameters
        custom_callables : dict < str, lambda >
            strings and lambda functions corresponding to some information we
            want to extract from the environment. The lambda will be called at
            each step to extract information from the env and it will be stored
            in a dict keyed by the str.
        """
        self.custom_callables = custom_callables or {}

        # Get the env name and a creator for the environment.
        create_env, env_name = make_create_env(flow_params)

        # record env_name and create_env, need it to register for ray
        self.env_name = env_name
        self.create_env = create_env

        # Create the environment.
        if not register_with_ray:
            self.env = create_env()

            logging.info(" Starting experiment {} at {}".format(
                self.env.network.name, str(datetime.utcnow())))

            logging.info("Initializing environment.")

    def run(self, num_runs, rl_actions=None, convert_to_csv=False, to_aws=None, only_query="", is_baseline=False,
            multiagent=False, rets=None, policy_map_fn=None):
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
        to_aws: str
            Specifies the S3 partition you want to store the output file,
            will be used to later for query. If NONE, won't upload output
            to S3.
        only_query: str
            Specifies which queries should be automatically run when the
            simulation data gets uploaded to S3. If an empty str is passed in,
            then it implies no queries should be run on this.
        is_baseline: bool
            Specifies whether this is a baseline run.

        Returns
        -------
        info_dict : dict < str, Any >
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

        # used to store
        info_dict = {
            "velocities": [],
            "outflows": [],
        }
        info_dict.update({
            key: [] for key in self.custom_callables.keys()
        })

        if rl_actions is None:
            def rl_actions(*_):
                return None

        # time profiling information
        t = time.time()
        times = []

        # data pipeline
        extra_info = defaultdict(lambda: [])
        source_id = 'flow_{}'.format(uuid.uuid4().hex)
        metadata = defaultdict(lambda: [])
        # collect current time
        cur_datetime = datetime.now(timezone.utc)
        cur_date = cur_datetime.date().isoformat()
        cur_time = cur_datetime.time().isoformat()
        # collecting information for metadata table
        metadata['source_id'].append(source_id)
        metadata['submission_time'].append(cur_time)
        metadata['network'].append(network_name_translate(self.env.network.name.split('_20')[0]))
        metadata['is_baseline'].append(str(is_baseline))
        name, strategy = get_configuration()
        metadata['submitter_name'].append(name)
        metadata['strategy'].append(strategy)

        if convert_to_csv and self.env.simulator == "traci":
            dir_path = self.env.sim_params.emission_path

        if dir_path:
            trajectory_table_path = os.path.join(dir_path, '{}.csv'.format(source_id))
            metadata_table_path = os.path.join(dir_path, '{}_METADATA.csv'.format(source_id))

        for i in range(num_runs):
            if rets and multiagent:
                ret = {key: [0] for key in rets.keys()}
            else:
                ret = 0
            vel = []
            custom_vals = {key: [] for key in self.custom_callables.keys()}
            run_id = "run_{}".format(i)
            self.env.pipeline_params = (extra_info, source_id, run_id)
            state = self.env.reset()
            for j in range(num_steps):
                t0 = time.time()
                state, reward, done, _ = self.env.step(rl_actions(state))
                t1 = time.time()
                times.append(1 / (t1 - t0))

                # Compute the velocity speeds and cumulative returns.
                veh_ids = self.env.k.vehicle.get_ids()
                vel.append(np.mean(self.env.k.vehicle.get_speed(veh_ids)))
                if rets and multiagent:
                    for actor, rew in reward.items():
                        ret[policy_map_fn(actor)][0] += rew
                elif not multiagent:
                    ret += reward

                # collect additional information for the data pipeline
                get_extra_info(self.env.k.vehicle, extra_info, veh_ids, source_id, run_id)

                # write to disk every 100 steps
                if convert_to_csv and self.env.simulator == "traci" and j % 100 == 0 and dir_path:
                    write_dict_to_csv(trajectory_table_path, extra_info, not j)
                    extra_info.clear()

                # Compute the results for the custom callables.
                for (key, lambda_func) in self.custom_callables.items():
                    custom_vals[key].append(lambda_func(self.env))

                if multiagent and done['__all__']:
                    break
                if type(done) is dict and done['__all__'] or type(done) is not dict and done:
                    break

            if rets and multiagent:
                for key in rets.keys():
                    rets[key].append(ret[key])
            elif not multiagent:
                rets.append(ret)

            # Store the information from the run in info_dict.
            outflow = self.env.k.vehicle.get_outflow_rate(int(500))
            if not multiagent:
                info_dict["returns"] = rets
            info_dict["velocities"].append(np.mean(vel))
            info_dict["outflows"].append(outflow)
            for key in custom_vals.keys():
                info_dict[key].append(np.mean(custom_vals[key]))

            if rets and multiagent:
                for agent_id, rew in rets.items():
                    print('Round {}, Return: {} for agent {}'.format(
                            i, ret, agent_id))
            elif not multiagent:
                print('Round {}, Return: {}'.format(i, ret))

        # Print the averages/std for all variables in the info_dict.
        for key in info_dict.keys():
            print("Average, std {}: {}, {}".format(
                key, np.mean(info_dict[key]), np.std(info_dict[key])))

        print("Total time:", time.time() - t)
        print("steps/second:", np.mean(times))
        self.env.terminate()

        if convert_to_csv and self.env.simulator == "traci":
            # wait a short period of time to ensure the xml file is readable
            time.sleep(0.1)

            write_dict_to_csv(trajectory_table_path, extra_info)
            write_dict_to_csv(metadata_table_path, metadata, True)

            if to_aws:
                upload_to_s3('circles.data.pipeline',
                             'metadata_table/date={0}/partition_name={1}_METADATA/{1}_METADATA.csv'.format(cur_date,
                                                                                                           source_id),
                             metadata_table_path)
                upload_to_s3('circles.data.pipeline',
                             'fact_vehicle_trace/date={0}/partition_name={1}/{1}.csv'.format(cur_date, source_id),
                             trajectory_table_path,
                             {'network': metadata['network'][0], 'is_baseline': metadata['is_baseline'][0]})

        return info_dict
