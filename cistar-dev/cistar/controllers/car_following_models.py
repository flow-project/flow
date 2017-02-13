import random

def makecfm(k_d=1, k_v=1, s=1):
    # k_d = proportional gain
    # k_v = derivative gain
    # s = safe distance

    def cfm(carID, env):
        leadID = env.get_leading_car(carID)
        print(carID, leadID)
        leadPos = env.get_x_by_id(leadID)
        leadVel = env.vehicles[leadID]['speed']

        thisPos = env.get_x_by_id(carID)
        thisVel = env.vehicles[carID]['speed']

        headway = (leadPos - thisPos) % env.scenario.length

        acc = k_d*(headway - s) + k_v*(leadVel - thisVel)
        return acc

    return cfm

def make_jank_cfm(k_d=1, k_v=1, s=1):
    # k_d = proportional gain
    # k_v = derivative gain
    # s = safe distance

    def cfm(carID, env):
        leadID = env.get_leading_car(carID)
        print(carID, leadID)
        leadPos = env.get_x_by_id(leadID)
        leadVel = env.vehicles[leadID]['speed']

        thisPos = env.get_x_by_id(carID)
        thisVel = env.vehicles[carID]['speed']

        headway = (leadPos - thisPos) % env.scenario.length

        acc = k_d*(headway - s) + k_v*(leadVel - thisVel)
        return acc + (random.random()-0.5)/5

    return cfm

def make_better_cfm(k_d=1, k_v=1, k_c = 1, d_des=1, v_des = 8):
    # k_d = proportional gain
    # k_v = derivative gain
    # s = safe distance

    def cfm(carID, env):
        leadID = env.get_leading_car(carID)
        leadPos = env.get_x_by_id(leadID)
        leadVel = env.vehicles[leadID]['speed']

        thisPos = env.get_x_by_id(carID)
        thisVel = env.vehicles[carID]['speed']

        d_l = (leadPos - thisPos) % env.scenario.length

        acc = k_d*(d_l - d_des) + k_v*(leadVel - thisVel) + k_c*(v_des - thisVel)
        return acc

    return cfm

def make_jank_bcm(k_d=1, k_v=1, s=1):
    # k_d = proportional gain
    # k_v = derivative gain
    # s = safe distance

    def bcm(carID, env):
        leadID = env.get_leading_car(carID)
        leadPos = env.get_x_by_id(leadID)
        leadVel = env.vehicles[leadID]['speed']

        thisPos = env.get_x_by_id(carID)
        thisVel = env.vehicles[carID]['speed']

        trailID = env.get_trailing_car(carID)
        trailPos = env.get_x_by_id(trailID)
        trailVel = env.vehicles[trailID]['speed']

        headway = (leadPos - thisPos) % env.scenario.length

        footway = (thisPos - trailPos) % env.scenario.length

        print(carID, headway, footway)

        acc = 0.5*k_d*((headway) - (footway)) + \
            0.5*k_v*((leadVel - thisVel) - (thisVel - trailVel))
        return acc

    return bcm

def make_better_bcm(k_d=1, k_v=1, k_c = 1, d_des=1, v_des = 8):
    # k_d = proportional gain
    # k_v = derivative gain

    def bcm(carID, env):
        leadID = env.get_leading_car(carID)
        leadPos = env.get_x_by_id(leadID)
        leadVel = env.vehicles[leadID]['speed']

        thisPos = env.get_x_by_id(carID)
        thisVel = env.vehicles[carID]['speed']

        trailID = env.get_trailing_car(carID)
        trailPos = env.get_x_by_id(trailID)
        trailVel = env.vehicles[trailID]['speed']

        headway = (leadPos - thisPos) % env.scenario.length # d_l

        footway = (thisPos - trailPos) % env.scenario.length # d_f

        print(carID, headway, footway)

        acc = k_d * (headway - footway) + \
            k_v * ((leadVel - thisVel) - (thisVel - trailVel)) + \
            k_c * (v_des - thisVel)

        # There would also be additional control rules that take
        # into account minimum safe separation, relative speeds,
        # speed limits, weather and lighting conditions, traffic density
        # and traffic advisories

        return acc

    return bcm
