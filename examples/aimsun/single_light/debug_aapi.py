import AAPI as aapi
import PyANGKernel as gk
from collections import OrderedDict
import random as r
# import numpy as np

model = gk.GKSystem.getSystem().getActiveModel()
global edge_detector_dict
edge_detector_dict = {}

target_nodes = [3329, 3344, 3370, 3341, 3369]
node_id = 3344
interval = 15*60


def random_sum_to(n, num_terms=None):
    num_terms = (num_terms) - 1
    a = r.sample(range(1, n), num_terms) + [0, n]
    list.sort(a)
    return a[i+1] - a[i] for i in range(len(a) - 1)


def get_current_phase(node_id):
    num_rings = aapi.ECIGetCurrentNbRingsJunction(node_id)
    num_phases = [0]*num_rings
    curr_phase = [None]*num_rings
    for ring_id in range(num_rings):
        num_phases[ring_id] = aapi.ECIGetNumberPhasesInRing(node_id, ring_id)
        curr_phase[ring_id] = aapi.ECIGetCurrentPhaseInRing(node_id, ring_id)
        if ring_id > 0:
            curr_phase[ring_id] += sum(num_phases[:ring_id])
    return curr_phase


def get_duration_phase(node_id, phase, timeSta):
    normalDurationP = aapi.doublep()
    maxDurationP = aapi.doublep()
    minDurationP = aapi.doublep()
    aapi.ECIGetDurationsPhase(node_id, phase, timeSta,
                              normalDurationP, maxDurationP, minDurationP)
    normalDuration = normalDurationP.value()
    maxDuration = maxDurationP.value()
    minDuration = minDurationP.value()
    return normalDuration, maxDuration, minDuration


def get_phases(ring_id):
    phases = []
    num_phases = aapi.ECIGetNumberPhasesInRing(node_id, ring_id)
    for phase in range(1, num_phases*2+1):
        phases.append(phase)
    return phases


def get_ids(node_id):
    # returns number of rings and control_id
    control_id = aapi.ECIGetNumberCurrentControl(node_id)
    num_rings = aapi.ECIGetNbRingsJunction(control_id, node_id)

    return control_id, num_rings


def get_total_green(node_id, ring_id, timeSta):
    # Format is set current control plan
    sum_interphase = 0
    control_id = aapi.ECIGetNumberCurrentControl(node_id)
    control_cycle = aapi.ECIGetControlCycleofJunction(control_id, node_id)
    num_phases = aapi.ECIGetNumberPhasesInRing(node_id, ring_id)
    for phase in (range(1, num_phases + 1)):
        if aapi.ECIIsAnInterPhase(node_id, phase, timeSta) == 0:
            continue
        else:
            _, _, duration = get_duration_phase(node_id, phase, timeSta)
            sum_interphase += duration

    total_green = control_cycle - sum_interphase
    return total_green


def change_phase_timing(node_id, timeSta):  # for debug_only
    # For Current Control Plan
    x = 0
    control_id = aapi.ECIGetNumberCurrentControl(node_id)  # Get control index
    num_rings = aapi.ECIGetNbRingsJunction(control_id, node_id)  # Get Number of rings
    for ring_id in range(num_rings):
        curr_phase = aapi.ECIGetCurrentPhaseInRing(node_id, ring_id)
        num_phases = aapi.ECIGetNumberPhasesInRing(node_id, ring_id)
        if curr_phase == num_phases:
            total_green = int(get_total_green(node_id, ring_id, timeSta))
            duration_list = random_sum_to(total_green, 4)  # randomized for debug only
            print(duration_list)
            a = 1
            if ring_id > 0:
                a = num_phases + 1
                num_phases = num_phases*2
            for phase in range(a, num_phases+1):
                if aapi.ECIIsAnInterPhase(node_id, phase, timeSta) == 1:
                    continue
                else:
                    duration = duration_list[x]
                    aapi.ECIChangeTimingPhase(node_id, phase, duration, timeSta)
                    x += 1
                    if x == 4:
                        x = 0
            duration = get_total_green(node_id, ring_id, timeSta)
            print('Ring ID: {}, Duration:{}'.format(ring_id, duration))
    return 0


def set_max_duration(control_id, node_id, phase, duration, timeSta):
    min_duration, max_duration, duration = get_duration_phase(node_id, phase, timeSta)
    if duration > int(max_duration):
        aapi.ECISetActuatedParamsMaxGreen(control_id, node_id, phase, duration)


def change_barrier_timing(node_id, timeSta):
    # For Current Control Plan
    x = 0
    control_id = aapi.ECIGetNumberCurrentControl(node_id)  # Get control index
    num_rings = aapi.ECIGetNbRingsJunction(control_id, node_id)  # Get Number of rings
    total_green = int(get_total_green(node_id, 0, timeSta))
    duration_list = random_sum_to(total_green, 4)  # randomized for debug only
    print(duration_list)
    for ring_id in range(num_rings):
        num_phases = aapi.ECIGetNumberPhasesInRing(node_id, ring_id)
        a = 1
        if ring_id > 0:
            a = num_phases + 1
            num_phases = num_phases*2
        for phase in range(a, num_phases+1):
            if aapi.ECIIsAnInterPhase(node_id, phase, timeSta) == 1:
                continue
            else:
                duration = duration_list[x]
                # set_max_min_duration(control_id, node_id, phase, duration, timeSta)
                aapi.ECIChangeTimingPhase(node_id, phase, duration, timeSta)
                x += 1
                if x == 4:
                    x = 0
        duration = get_total_green(node_id, ring_id, timeSta)
        print('Ring ID: {}, Duration:{}'.format(ring_id, duration))
    return 0


def AAPILoad():
    return 0


def AAPIInit():
    return 0


def AAPIManage(time, timeSta, timeTrans, acycle):
    return 0


def AAPIPostManage(time, timeSta, timeTrans, acycle):
    if time % 900 == 0:
        control_id, ring_id, phase_list = get_current_ids(node_id)

    return 0


def AAPIFinish():
    """ring_id = 0
    phase = aapi.ECIGetCurrentPhaseInRing(node_id, ring_id)
    print(phase)
    normalDurationP = aapi.doublep()
    maxDurationP = aapi.doublep()
    minDurationP = aapi.doublep()
    aapi.ECIGetDurationsPhase(node_id, phase, timeSta,
                                normalDurationP, maxDurationP, minDurationP)
    normalDuration = normalDurationP.value()
    maxDuration = maxDurationP.value()
    minDuration = minDurationP.value()
    print('normalDuration: {} maxDuration: {} minDuration: {}'.format(normalDuration, maxDuration,minDuration))
    # thus I that maxDuration, minDuration == MinGreen and MaxOut
    # So where should i put thiisssss, I think probably in the RL training? Because it is a condition for the predicted value!
    # Hmm, sounds like a great hypothesis. Should consult others though
    # Therefore, let's go create an environment!!!!!!"""
    return 0


def AAPIUnLoad():
    return 0


def AAPIPreRouteChoiceCalculation(time, timeSta):
    return 0


def AAPIEnterVehicle(idveh, idsection):
    return 0


def AAPIExitVehicle(idveh, idsection):
    return 0


def AAPIEnterPedestrian(idPedestrian, originCentroid):
    return 0


def AAPIExitPedestrian(idPedestrian, destinationCentroid):
    return 0


def AAPIEnterVehicleSection(idveh, idsection, atime):
    return 0


def AAPIExitVehicleSection(idveh, idsection, atime):
    return 0
