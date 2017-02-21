

def changeFasterLaneBuilder(speedThreshold = 5, likelihood_mult = 0.5,
                            dxBack = 0, dxForward = 60,
                            gapBack = 10, gapForward = 5):
    """
    Intelligent lane changer
    :param speedThreshold: minimum speed increase required
    :param likelihood_mult: probability change will be requested if warranted = this * speedFactor
    :param dxBack: Farthest distance back car can see
    :param dxForward: Farthest distance forward car can see
    :param gapBack: Minimum required clearance behind car
    :param gapForward: Minimum required clearance in front car
    :return: carFn to input to a carParams
    """
    def carFn(carID, env):
        num_lanes = env.scenario.lanes
        v = [0] * env.scenario.lanes
        for lane in range(.numLanes):
            if sim.getCars(idx, dxBack=gapBack, dxForward=gapForward, lane=lane):
                # cars too close, no lane changing allowed
                v[lane] = 0
                continue
            cars = sim.getCars(idx, dxBack=dxBack, dxForward=dxForward, lane=lane)
            if len(cars) > 0:
                v[lane] = mean([c["v"] for c in cars])
            else:
                v[lane] = traci.vehicle.getMaxSpeed(car["id"])
        maxv = max(v)
        maxl = v.index(maxv)
        myv = v[car["lane"]]

        if maxl != car["lane"] and \
           (maxv - myv) > speedThreshold and \
           random.random() < likelihood_mult * car["f"]:
            traci.vehicle.changeLane(car["id"], maxl, 10000)
    return carFn