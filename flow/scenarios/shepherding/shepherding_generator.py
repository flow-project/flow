from flow.scenarios.loop.gen import CircleGenerator
from flow.core.util import printxml
from flow.core.util import makexml
from lxml import etree
import random

E = etree.Element

class ShepherdingGenerator(CircleGenerator):
    def make_routes(self, scenario, initial_config):

        vehicles = scenario.vehicles
        if vehicles.num_vehicles > 0:
            routes = makexml("routes", "http://sumo.dlr.de/xsd/routes_file.xsd")

            # add the types of vehicles to the xml file
            for type, type_params in vehicles.types:
                type_params_str = {key: str(type_params[key]) for key in type_params}
                routes.append(E("vType", id=type, **type_params_str))

            self.vehicle_ids = vehicles.get_ids()

            if initial_config.shuffle:
                random.shuffle(self.vehicle_ids)

            # add the initial positions of vehicles to the xml file
            positions = initial_config.positions
            lanes = initial_config.lanes
            for i, id in enumerate(scenario.vehicles.get_sumo_ids()):
                veh_type = vehicles.get_state(id, "type")
                edge, pos = positions[i]
                lane = lanes[i]
                type_depart_speed = vehicles.get_initial_speed(id)
                routes.append(self._vehicle(
                    veh_type, "route" + edge, depart="0", id=id,
                    color="1,0.0,0.0", departSpeed=str(type_depart_speed),
                    departPos=str(pos), departLane=str(lane)))

            positions = [('bottom', 305), ('bottom', 305), ('bottom', 305)]
            lanes = [0,1,2]
            for i, id in enumerate(["rl_0", "rl_1", "rl_2"]):
                veh_type = vehicles.get_state(id, "type")
                edge, pos = positions[i]
                lane = lanes[i]
                type_depart_speed = vehicles.get_initial_speed(id)
                routes.append(self._vehicle(
                    veh_type, "route" + edge, depart="0", id=id,
                    color="1,0.0,0.0", departSpeed=str(type_depart_speed),
                    departPos=str(pos), departLane=str(lane)))

            printxml(routes, self.cfg_path + self.roufn)