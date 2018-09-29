class Vehicle:

    # Implemented assuming kernel is Traci for now

    def __init__(self, kernel):
        self.K = kernel # Assume is Traci for now


    def get_ids(self):
        # Returns a list of all objects in the network
        return self.K.vehicle.getIDList()


    def get_human_ids(self, vehID):
        # Returns the list of persons which includes those defined using attribute
        # 'personNumber' as well as persons riding this vehicle
        return self.K.vehicle.getPersonIDList(vehID)


    def get_controlled_ids(self):
        raise NotImplementedError


    def get_controlled_lc_ids(self):
        raise NotImplementedError


    def get_rl_ids(self):
        raise NotImplementedError


    def get_speed(self, vehID):
        # Returns speed in m/s of named vehicle within the last step
        return self.K.vehicle.getSpeed(vehID)


    def get_absolute_position(self):
        raise NotImplementedError


    def get_position(self, vehID):
        # Returns a positon tuple of named vehicle within last step
        return self.K.vehicle.getPosition(vehID)


    def get_edge(self, vehID):
        # Returns the id of the edge the named vehicle was at within last step
        return self.K.vehicle.getRoadID(vehID)


    def get_lane(self, vehID):
        # Returns the id of the lane the named vehicle was at within last step
        return self.K.vehicle.getLaneID(vehID)


    def get_length(self, vehID):
        # Returns the length in m of the given vehicle
        return self.K.vehicle.getLength(vehID)


    def get_route(self, vehID):
        # Returns a list of the ids of edges the vehicle's route is made of
        return self.K.vehicle.getRoute(vehID)


    def get_leader(self, vehID, dist=0.0):
        #  Returns a tuple of the (leading vehicle id, distance). Distance is measured
        # from front + minGap to the back of leader, so it does not include the minGap
        # of the vehicle. The dist parameter defines the maximum lookahead, where 0.0
        # calculates a lookahead from the brake gap. Note that the returned leader may
        # be further away than the passed in dist
        return self.K.vehicle.getLeader(vehID, dist)


    def get_follower(self, vehID, dist=0.0):
        # Because Traci has no explicit get_follower method, searches through all vehicles 
        # and returns a tuple of (follower vehicle ID, distance) of the first vehicle 
        # whose leader matches the given vehicle. Distance is measured from the front 
        # + minGap to the back of leader.
        for vehID2 in self.get_ids():
            curr_lead, curr_dist = self.K.vehicle.getLeader(vehID2, dist)
            if curr_lead == vehID:
                return (vehID2, curr_dist) 

        # If no leader is found, empty string ID is returned
        return ("", 0.0)


    def get_headway(self, vehID):
        # Returns the driver's reaction time in s for given vehicle
        return self.K.vehicle.getTau(vehID)


    def get_lane_headways(self):
        raise NotImplementedError


    def get_lane_tailways(self):
        raise NotImplementedError


    def get_lane_leaders(self):
        raise NotImplementedError


    def get_lane_followers(self):
        raise NotImplementedError


    def get_acc_controller(self):
        # Maybe make this hidden
        raise NotImplementedError


    def get_lane_changing_controller(self):
        # Maybe make this hidden
        raise NotImplementedError


    def get_routing_controller(self):
        # Maybe make this hideen
        raise NotImplementedError


    def get_inflow_rate(self):
        # Maybe move to scenario subclass instead
        raise NotImplementedError


    def get_outflow_rate(self):
        # Maybe move to scenario subclass instead
        raise NotImplementedError


    def get_initial_speed(self):
        # Seems sumo-specific
        raise NotImplementedError


    def get_lane_change_mode(self):
        # Seems sumo-specific
        raise NotImplementedError


    def get_speed_mode(self):
        # Seems sumo-specific
        raise NotImplementedError


    def get_num_arrived(self):
        # Seems sumo-specific
        raise NotImplementedError


    def apply_acceleration(self, vehID, accel):
        # Sets the maximum acceleration in m/s^2 of given vehicle
        return self.K.vehicle.setAccel(vehID, accel)


    def apply_lane_change(self, vehID, laneIndex, duration):
        # Forces a lane change of given vehicle to given lane. If
        # successful, lane will be chosen for given time duration
        # (in ms).
        return self.K.vehicle.changeLane(vehID, laneIndex, duration)


    def choose_route(self, vehID, edgeList):
        # Changes the vehicle's route to the given list of edges.
        # First edge in the list has to be the one the vehicle is at
        # in the moment.
        return self.K.vehicle.setRoute(vehID, edgeList)




