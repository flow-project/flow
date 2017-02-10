def cfm(lead, thisCar):
    leadPos, leadVel = lead[0], lead[1]
    thisPos, thisVel = thisCar[0], thisCar[1]

    k_d = 1 # proportional gain
    k_v = 1 # derivative gain
    s = 1 # safe distance

    acc = k_d*(leadPos - thisCarPos - s) + k_v*(leadVel - thisVel)
    return acc

def bcm(lead, thisCar, follow):
    leadPos, leadVel = lead[0], lead[1]
    thisPos, thisVel = thisCar[0], thisCar[1]
    followPos, followVel = follow[0], follow[1]

    k_d = 1 # proportional gain
    k_v = 1 # derivative gain
    s = 1 # safe distance

    acc = 0.5*k_d*((leadPos - thisCarPos) - (thisPos - followPos)) + \
        0.5*k_v*((leadVel - thisVel) - (thisVel - followVel))
    return acc
