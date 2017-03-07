import traci




class base_vehicle:
    """
    THIS CLASS IS NOT CURRENTLY BEING USED :^)
    To be implemented soon
    """
    def __init__(self, veh_id):
        self.veh_id = veh_id
        self.type = traci.vehicle.getTypeID(veh_id)
        self.edge = traci.vehicle.getRoadID(veh_id)
        self.position = traci.vehicle.getLanePosition(veh_id)
        self.lane = traci.vehicle.getLaneIndex(veh_id)
        self.speed = traci.vehicle.getSpeed(veh_id)
        self.length = traci.vehicle.getLength(veh_id)
        self.max_speed = traci.vehicle.getMaxSpeed(veh_id)
        self.traci.vehicle.setSpeedMode(veh_id, 0)

    def create_controller(self):
        raise NotImplementedError

    def update(self):
        self.edge = traci.vehicle.getRoadID(self.veh_id)
        self.position = traci.vehicle.getLanePosition(self.veh_id)
        self.lane = traci.vehicle.getLaneIndex(self.veh_id)
        self.speed = traci.vehicle.getSpeed(self.veh_id)

    def reset(self):
        self.type = traci.vehicle.getTypeID(self.veh_id)
        self.edge = traci.vehicle.getRoadID(self.veh_id)
        self.position = traci.vehicle.getLanePosition(self.veh_id)
        self.lane = traci.vehicle.getLaneIndex(self.veh_id)
        self.speed = traci.vehicle.getSpeed(self.veh_id)