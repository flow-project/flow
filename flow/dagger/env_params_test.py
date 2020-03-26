from flow.networks.ring import RingNetwork
name = "ring_example"

from flow.core.params import VehicleParams
vehicles = VehicleParams()

from flow.controllers.car_following_models import IDMController
from flow.controllers.routing_controllers import ContinuousRouter
from imitating_controller import ImitatingController
vehicles.add("human",
             acceleration_controller=(IDMController, {}),
             routing_controller=(ContinuousRouter, {}),
             num_vehicles=22)

from flow.networks.ring import ADDITIONAL_NET_PARAMS
from flow.core.params import NetParams
net_params = NetParams(additional_params=ADDITIONAL_NET_PARAMS)

from flow.core.params import InitialConfig
initial_config = InitialConfig(spacing="uniform", perturbation=1)

from flow.core.params import TrafficLightParams
traffic_lights = TrafficLightParams()

from flow.envs.ring.accel import AccelEnv
from flow.core.params import SumoParams
sim_params = SumoParams(sim_step=0.1, render=False, emission_path='data')

from flow.envs.ring.accel import ADDITIONAL_ENV_PARAMS
from flow.core.params import EnvParams
env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

flow_params_test = dict(
    exp_tag='ring_example',
    env_name=AccelEnv,
    network=RingNetwork,
    simulator='traci',
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
    initial=initial_config,
    tls=traffic_lights,
)

# number of time steps
flow_params_test['env'].horizon = 3000
