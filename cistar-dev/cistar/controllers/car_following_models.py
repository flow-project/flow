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


def makebcm(k_d=1, k_v=1, s=1):
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
