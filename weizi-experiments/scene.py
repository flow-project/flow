

'''
"SimpleGridScenario",
"HighwayScenario", "LoopScenario", "MergeScenario",
"TwoLoopsOneMergingScenario", "MultiLoopScenario", "MiniCityScenario",
"HighwayRampsScenario"
'''


from flow.core.params import VehicleParams
vehicles = VehicleParams()

from flow.controllers import IDMController, ContinuousRouter
vehicles.add(
	veh_id="idm",
	acceleration_controller=(IDMController, {}),
	routing_controller=(ContinuousRouter, {}),
	num_vehicles=8)

from flow.core.params import NetParams, InitialConfig
initial_config = InitialConfig()



from flow.scenarios.multi_loop import MultiLoopScenario, ADDITIONAL_NET_PARAMS
scenario = MultiLoopScenario(
	name = 'test_scenario',
	vehicles = vehicles,
	net_params = NetParams(additional_params=ADDITIONAL_NET_PARAMS.copy()),
	initial_config=initial_config
)


from flow.core.params import EnvParams
from flow.envs.loop.loop_accel import ADDITIONAL_ENV_PARAMS
env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

from flow.core.params import SumoParams
sim_params = SumoParams(sim_step=0.1, render=True)

from flow.envs.loop.loop_accel import AccelEnv
env = AccelEnv(env_params, sim_params, scenario)

from flow.core.experiment import Experiment
exp = Experiment(env)
exp.run(1, 3000)

