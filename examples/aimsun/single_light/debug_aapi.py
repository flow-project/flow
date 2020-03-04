import csv
import AAPI as aapi
import PyANGKernel as gk
from collections import OrderedDict
import random as r
import sys
sys.path.append('/home/cjrsantos/anaconda3/envs/aimsun_flow/lib/python2.7/site-packages')
import numpy as np

model = gk.GKSystem.getSystem().getActiveModel()
global edge_detector_dict
edge_detector_dict = {}

target_nodes = [3329, 3344, 3370, 3341, 3369]
westbound_section = [506, 563, 24660, 568, 462]
eastbound_section = [338, 400, 461, 24650, 450]
node_id = 3344
interval = 15*60
carPos = 0

with open('test.csv', 'w') as csvFile:
    data = []
    fieldnames = ['centroid']
    csv_writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
    csv_writer.writeheader()


def random_sum_to(n, num_terms=None):
    num_terms = (num_terms) - 1
    a = r.sample(range(1, n), num_terms) + [0, n]
    list.sort(a)
    return [a[i+1] - a[i] for i in range(len(a) - 1)]


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


def enter_traffic_flow(time):
    sections = []

    num_sections = aapi.AKIInfNetNbSectionsANG()

    for section in range(num_sections):
        sections.append(aapi.AKIInfNetGetSectionANGId(section))

    rand_sections = np.random.choice(sections, 100)

    for section in rand_sections:
        aapi.AKIEnterVehTrafficFlow(section, carPos, 0)
    return 0


def get_centroid(node_id):
    centroid_list = []
    num_centroids = aapi.AKIInfNetNbCentroids()
    # print(num_centroids)
    for i in range(num_centroids):
        centroid_list.append(aapi.AKIInfNetGetCentroidId(i))
    print(centroid_list)

    centroid_origin = [8040650, 8040653, 8040655, 8040657, 8040659, 8040661, 8040663, 8040665, 8040667, 8040669, 8040671,
                       8040673, 8040675, 8040677, 8040679, 8040682, 8040688, 8040690, 8040696, 8040699, 8040701, 8040705,
                       8040707, 8040710]
    centroid_destination = [8040677, 8040679, 8040682, 8040688, 8040690, 8040696, 8040699, 8040701, 8040705,
                            8040707, 8040710, 8040712, 8040714, 8040716, 8040718, 8040720, 8040722, 8040724, 8040726, 8040728,
                            8040730, 8040732, ]
    i = 0
    while i < 6:
        #origin = np.random.choice(centroid_origin)
        origin = centroid_origin[0]
        #destination = np.random.choice(centroid_destination)
        destination = centroid_destination[0]

        new_demand = 150
        aapi.AKIODDemandSetDemandODPair(origin, destination, 1, 0, new_demand)
        stat = aapi.AKIODDemandSetDemandODPair(origin, destination, 1, 0, new_demand)

        demand = aapi.AKIODDemandGetDemandODPair(origin, destination, 1, 0)
        print(origin, destination, demand, stat)
        i += 1


def AAPILoad():
    return 0


def AAPIInit():
    model = gk.GKSystem.getSystem().getActiveModel()
    warmup_time = gk.GKExperiment.getWarmUpTime().getActiveModel()
    print(model, warmup_time)

    return 0
    """for section in westbound_section:
        flow = aapi.AKIStateDemandGetDemandSection(section, 0, 0) 
        #aapi.AKIStateDemandSetDemandSection(section, 0, 1, double anewflow)
        print(section, flow) 
    return 0"""


def AAPIManage(time, timeSta, timeTrans, acycle):
    
    if time % 60 == 0:
        for section in westbound_section:
            flow = aapi.AKIStateDemandGetDemandSection(section, 1, 0)
            # aapi.AKIStateDemandSetDemandSection(section, 0, 1, double anewflow)
            print(section, flow)
    return 0


def AAPIPostManage(time, timeSta, timeTrans, acycle):
    # if time % 900 == 0:
        #control_id, ring_id, phase_list = get_current_ids(node_id)

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
