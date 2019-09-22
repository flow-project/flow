"""Repeatedly opens up a sumo port to test for race conditions."""
from flow.controllers import IDMController, ContinuousRouter
from flow.core.params import SumoParams, EnvParams, \
    InitialConfig, NetParams
from flow.core.params import VehicleParams
from flow.envs.ring.accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.networks.ring import RingNetwork, ADDITIONAL_NET_PARAMS
import ray


@ray.remote
def start():
    """Start a environment object with ray."""
    sim_params = SumoParams(sim_step=0.1, render=False)

    vehicles = VehicleParams()
    vehicles.add(
        veh_id="idm",
        acceleration_controller=(IDMController, {}),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=22)

    env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

    additional_net_params = ADDITIONAL_NET_PARAMS.copy()
    net_params = NetParams(additional_params=additional_net_params)

    initial_config = InitialConfig(bunching=20)

    network = RingNetwork(
        name="ring",
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config)

    env = AccelEnv(env_params, sim_params, network)
    env.close()


ray.init()
results = ray.get([start.remote() for i in range(10000)])
