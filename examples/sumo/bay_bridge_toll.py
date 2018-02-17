"""
Example of a multi-lane network with human-driven vehicles.
"""
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, \
    InFlows
from flow.core.vehicles import Vehicles
from flow.core.traffic_lights import TrafficLights

from flow.core.experiment import SumoExperiment
from flow.envs.bottleneck_env import BottleneckEnv
from flow.scenarios.bridge_toll.gen import BBTollGenerator
from flow.scenarios.bridge_toll.scenario import BBTollScenario
from flow.controllers.lane_change_controllers import *

NUM_LANES = 16  # number of lanes in the widest highway

logging.basicConfig(level=logging.INFO)

sumo_params = SumoParams(sumo_binary="sumo-gui")

vehicles = Vehicles()
vehicles.add(veh_id="human",
             speed_mode=0b11111,
             lane_change_mode=1612,
             num_vehicles=100)

additional_env_params = {"target_velocity": 8}
env_params = EnvParams(additional_params=additional_env_params)

# flow rate
flow_rate = 15000
# percentage of flow coming out of each lane
flow_dist = np.random.dirichlet(np.ones(NUM_LANES), size=1)[0]

inflow = InFlows()
for i in range(NUM_LANES):
    lane_num = str(i)
    veh_per_hour = flow_rate * flow_dist[i]
    print(veh_per_hour)
    inflow.add(veh_type="human", edge="1", vehsPerHour=veh_per_hour,
               departLane=lane_num, departSpeed=10)

traffic_lights = TrafficLights()
traffic_lights.add(node_id="2")
traffic_lights.add(node_id="3")

net_params = NetParams(in_flows=inflow,
                       no_internal_links=False)

initial_config = InitialConfig(spacing="random", min_gap=5,
                               lanes_distribution=float("inf"),
                               edges_distribution=["1", "2", "3", "4"])

scenario = BBTollScenario(name="bay_bridge_toll",
                          generator_class=BBTollGenerator,
                          vehicles=vehicles,
                          net_params=net_params,
                          initial_config=initial_config,
                          traffic_lights=traffic_lights)

env = BottleneckEnv(env_params, sumo_params, scenario)

exp = SumoExperiment(env, scenario)

logging.info("Experiment Set Up complete")

exp.run(1, 1500)
