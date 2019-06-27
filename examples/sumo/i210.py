from flow.core.experiment import Experiment
from flow.core.params import SumoParams, EnvParams, \
    InitialConfig, NetParams
from flow.core.params import VehicleParams
from flow.envs.test import TestEnv
from flow.scenarios.i210 import I210Scenario


if __name__ == "__main__":
    scenario = I210Scenario(
        name="i210",
        vehicles=VehicleParams(),
        net_params=NetParams(),
        initial_config=InitialConfig())

    env = TestEnv(EnvParams(), SumoParams(render=True), scenario)

    exp = Experiment(env)
    exp.run(1, 10)
