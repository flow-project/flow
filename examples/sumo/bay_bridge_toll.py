"""
Example of a multi-lane network with human-driven vehicles.
"""
import logging
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, InFlows
from flow.controllers.routing_controllers import *
from flow.core.vehicles import Vehicles

from flow.core.experiment import SumoExperiment
from flow.envs.loop_accel import AccelEnv
from flow.scenarios.bridge_toll.gen import BBTollGenerator
from flow.scenarios.bridge_toll.scenario import BBTollScenario
from flow.controllers.car_following_models import *
from flow.controllers.lane_change_controllers import *

logging.basicConfig(level=logging.INFO)

sumo_params = SumoParams(sumo_binary="sumo-gui")

vehicles = Vehicles()
vehicles.add(veh_id="human",
             lane_change_controller=(StaticLaneChanger, {}),
             initial_speed=0,
             num_vehicles=40)

additional_env_params = {"target_velocity": 8}
env_params = EnvParams(additional_params=additional_env_params)

inflow = InFlows()
inflow.add(veh_type="human", edge="1", vehsPerHour=15000/16,
           departLane="0", departSpeed=20)
inflow.add(veh_type="human", edge="1", vehsPerHour=15000/16,
           departLane="1", departSpeed=20)
inflow.add(veh_type="human", edge="1", vehsPerHour=15000/16,
           departLane="2", departSpeed=20)
inflow.add(veh_type="human", edge="1", vehsPerHour=15000/16,
           departLane="3", departSpeed=20)
inflow.add(veh_type="human", edge="1", vehsPerHour=15000/16,
           departLane="4", departSpeed=20)
inflow.add(veh_type="human", edge="1", vehsPerHour=15000/16,
           departLane="5", departSpeed=20)
inflow.add(veh_type="human", edge="1", vehsPerHour=15000/16,
           departLane="6", departSpeed=20)
inflow.add(veh_type="human", edge="1", vehsPerHour=15000/16,
           departLane="7", departSpeed=20)
inflow.add(veh_type="human", edge="1", vehsPerHour=15000/16,
           departLane="8", departSpeed=20)
inflow.add(veh_type="human", edge="1", vehsPerHour=15000/16,
           departLane="9", departSpeed=20)
inflow.add(veh_type="human", edge="1", vehsPerHour=15000/16,
           departLane="10", departSpeed=20)
inflow.add(veh_type="human", edge="1", vehsPerHour=15000/16,
           departLane="11", departSpeed=20)
inflow.add(veh_type="human", edge="1", vehsPerHour=15000/16,
           departLane="12", departSpeed=20)
inflow.add(veh_type="human", edge="1", vehsPerHour=15000/16,
           departLane="13", departSpeed=20)
inflow.add(veh_type="human", edge="1", vehsPerHour=15000/16,
           departLane="14", departSpeed=20)
inflow.add(veh_type="human", edge="1", vehsPerHour=15000/16,
           departLane="15", departSpeed=20)

net_params = NetParams(in_flows=inflow,
                       no_internal_links=False)

initial_config = InitialConfig(spacing="random",
                               min_gap=5, lanes_distribution=1000)

scenario = BBTollScenario(name="bay_bridge_toll",
                          generator_class=BBTollGenerator,
                          vehicles=vehicles,
                          net_params=net_params,
                          initial_config=initial_config)

env = AccelEnv(env_params, sumo_params, scenario)

exp = SumoExperiment(env, scenario)

logging.info("Experiment Set Up complete")

exp.run(1, 1500)
