import sys
# sys.path.append('/home/yashar/Aimsun_Next_8_3_0/programming/Aimsun Next API/AAPIPython/Micro')
import AAPI as aimsun_api

from flow.utils.rllib import get_flow_params
import json


with open('/home/yashar/git_clone/flow/flow/core/kernel/simulation/init.json') as f:
    data = json.load(f)

flow_params = get_flow_params(data)

exp_tag = flow_params['exp_tag']
net_params = flow_params['net']
vehicles = flow_params['veh']
initial_config = flow_params['initial']
module = __import__('flow.scenarios', fromlist=[flow_params['scenario']])
scenario_class = getattr(module, flow_params['scenario'])

scenario = scenario_class(
    name=exp_tag,
    vehicles=vehicles,
    net_params=net_params,
    initial_config=initial_config)

# Start the environment with the gui turned on and a path for the
# emission file
module = __import__('flow.envs', fromlist=[flow_params['env_name']])
env_class = getattr(module, flow_params['env_name'])
env_params = flow_params['env']
sumo_params = flow_params['sumo']

env = env_class(
    env_params=env_params,
    sumo_params=sumo_params,
    scenario=scenario
)


def AAPILoad():
    aimsun_api.AKIPrintString("AAPILoad")
    return 0


def AAPIInit():
    aimsun_api.AKIPrintString("AAPIInit")
    # TODO; trigger the environment
    return 0


def AAPIManage(time, timeSta, timeTrans, acycle):
    step_done = False
    while not step_done:
        # read tcp/ip commands
        message = tcp.read()

        if message is not None:
            message_type, message_content = message
            # check if it is an add vehicle message
            if message_type == ac.ADD_VEHICLE:
                edge_aimsun_id, lane, type_id, pos, speed, next_section, tracking = message_content
                # add vehicle in Aimsun
                next_section = -1  # negative one means the first feasible turn #TODO get route
                tracking = 1  # 1 if tracked, 0 otherwise
                type_id = 1
                id = AKIPutVehTrafficFlow(edge_aimsun_id, lane, type_id, pos, speed, next_section, tracking)



    time.wait()  # wait until you receive a message from simulation_step()
    env.step(rl_actions=None)
    aimsun_api.AKIPrintString("AAPIManage", time)
    return 0


def AAPIPostManage(time, timeSta, timeTrans, acycle):
    aimsun_api.AKIPrintString("AAPIPostManage")
    return 0


def AAPIFinish():

    AKIPrintString("AAPIFinish")
    return 0


def AAPIUnLoad():
    # AKIPrintString("AAPIUnLoad")
    return 0


def AAPIPreRouteChoiceCalculation(time, timeSta):
    # AKIPrintString("AAPIPreRouteChoiceCalculation")
    return 0


def AAPIEnterVehicle(idveh, idsection):
    return 0


def AAPIExitVehicle(idveh, idsection):
    return 0


def AAPIEnterPedestrian(idPedestrian, originCentroid):
    return 0


def AAPIExitPedestrian(idPedestrian, destinationCentroid):
    return 0


def AAPIEnterVehicleSection(idveh, idsection, atime):
    return 0


def AAPIExitVehicleSection(idveh, idsection, atime):
    return 0


if __name__ == '__main__':
    from flow.controllers import IDMController, ContinuousRouter
    from flow.core.experiment import SumoExperiment
    from flow.core.params import SumoParams, EnvParams, \
        InitialConfig, NetParams
    from flow.core.params import Vehicles
    from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
    from flow.scenarios.loop import LoopScenario, ADDITIONAL_NET_PARAMS


    def sugiyama_example(render=None):
        """
        Perform a simulation of vehicles on a ring road.

        Parameters
        ----------
        render : bool, optional
            specifies whether to use sumo's gui during execution

        Returns
        -------
        exp: flow.core.SumoExperiment type
            A non-rl experiment demonstrating the performance of human-driven
            vehicles on a ring road.
        """
        sumo_params = SumoParams(sim_step=0.1, render=True)

        if render is not None:
            sumo_params.render = render

        vehicles = Vehicles()
        vehicles.add(
            veh_id="idm",
            acceleration_controller=(IDMController, {}),
            routing_controller=(ContinuousRouter, {}),
            num_vehicles=22)

        env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

        additional_net_params = ADDITIONAL_NET_PARAMS.copy()
        net_params = NetParams(additional_params=additional_net_params)

        initial_config = InitialConfig(bunching=20)

        scenario = LoopScenario(
            name="sugiyama",
            vehicles=vehicles,
            net_params=net_params,
            initial_config=initial_config)

        env = AccelEnv(env_params, sumo_params, scenario)

        return SumoExperiment(env)


    if __name__ == "__main__":
        # import the experiment variable
        exp = sugiyama_example()

        # run for a set number of rollouts / time steps
        exp.run(1, 1500)
