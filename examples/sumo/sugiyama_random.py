"""Used as an example of sugiyama experiment.

This experiment consists of 21 IDM cars on a ring creating shockwaves, 1 RL vehicle.

Random policies are collected for training MDN-RNN for world models research.

"""

from flow.controllers import RLController, IDMController, ContinuousRouter
from flow.core.experiment import Experiment
from flow.core.params import SumoParams, EnvParams, \
    InitialConfig, NetParams
from flow.core.params import VehicleParams
from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.scenarios.loop import LoopScenario, ADDITIONAL_NET_PARAMS
import numpy as np

def sugiyama_example(render=None, rollout=0):
    """
    Perform a simulation of vehicles on a ring road.

    Parameters
    ----------
    render : bool, optional
        specifies whether to use the gui during execution

    Returns
    -------
    exp: flow.core.experiment.Experiment
        A non-rl experiment demonstrating the performance of human-driven
        vehicles on a ring road.
    """
    sim_params = SumoParams(sim_step=0.1, render=True, rollout=rollout)

    if render is not None:
        sim_params.render = render

    vehicles = VehicleParams()
    vehicles.add(
        veh_id="idm",
        acceleration_controller=(IDMController, {}),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=21)
    vehicles.add(
        veh_id="rl",
        acceleration_controller=(RLController, {}),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=1)

    env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

    additional_net_params = ADDITIONAL_NET_PARAMS.copy()
    net_params = NetParams(additional_params=additional_net_params)

    initial_config = InitialConfig(bunching=20)

    scenario = LoopScenario(
        name="sugiyama",
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config)

    env = AccelEnv(env_params, sim_params, scenario)

    return Experiment(env)


def make_random_rl_actions(horizon, rollout):
    """
    pre-generates a sequence of random rl actions beforehand
    samples a low and a high bound so the average speed of the RL controller moves
    :param horizon: number of time steps per rollout
    :return: rl_actions method
    """

    # sample low and high bounds to vary random actions by
    avg = np.random.uniform(0.3, 0.5)
    width = 0.5
    low = avg - width
    high = avg + width

    actions = np.random.uniform(low, high, horizon)
    filename = "snapshots2/sugiyama_rand_actions/roll%dlength%dactions%d" % (rollout, horizon, 1)
    np.save(filename, actions)

    step = -1

    def rl_actions(state):
        """
        from AccelEnv#get_state

        def get_state(self):
            speed = [self.k.vehicle.get_speed(veh_id) / self.k.scenario.max_speed()
                     for veh_id in self.sorted_ids]
            pos = [self.k.vehicle.get_x_by_id(veh_id) / self.k.scenario.length()
                   for veh_id in self.sorted_ids]
            return np.array(speed + pos)

        use this to return a random action
        :param state: list of speeds concatenated by actions
        :return: action
        """
        # print("state: " + str(state))
        nonlocal step
        step += 1
        return actions[step]

    return rl_actions

if __name__ == "__main__":
    horizon = 1500
    np.random.seed(123456)
    for roll in range(30):
        print("sampling rollout %d" % roll)
        # import the experiment variable
        exp = sugiyama_example(rollout=roll)

        # make a random rl action generator
        rl_actions = make_random_rl_actions(horizon, roll)

        # run for a set number of rollouts / time steps
        exp.run(1, horizon, rl_actions=rl_actions)
