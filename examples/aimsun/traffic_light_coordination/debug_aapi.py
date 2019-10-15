import AAPI as aapi
from PyANGKernel import *
from collections import OrderedDict

model = GKSystem.getSystem().getActiveModel()
global edge_detector_dict
edge_detector_dict = {}


def get_section_occupancy(section_id):
    section_data = edge_detector_dict[section_id]
    readings = [0, 0]
    detector_types = ["stopbar_ids", "advanced_ids"]
    for i, detector_type in enumerate(detector_types):
        for detector_id in section_data[detector_type]:
            reading = aapi.AKIDetGetTimeOccupedAggregatedbyId(detector_id, 0)
            readings[i] += max(reading, 0)/100  # percentage to float
    return readings


def get_section_flow(section_id):
    section_data = edge_detector_dict[section_id]
    readings = [0, 0]
    detector_types = ["stopbar_ids", "advanced_ids"]
    for i, detector_type in enumerate(detector_types):
        for detector_id in section_data[detector_type]:
            reading = aapi.AKIDetGetCounterAggregatedbyId(detector_id, 0)
            readings[i] += max(reading, 0)
    return readings


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


def get_num_phases(node_id):
    num_rings = aapi.ECIGetCurrentNbRingsJunction(node_id)
    return [aapi.ECIGetNumberPhasesInRing(node_id, i)
            for i in range(num_rings)]


def get_duration_phase(node_id, phase, timeSta):
    normalDurationP = aapi.doublep()
    maxDurationP = aapi.doublep()
    minDurationP = aapi.doublep()
    aapi.ECIGetDurationsPhase(node_id, phase, timeSta,
                              normalDurationP, maxDurationP, minDurationP)
    normalDuration = normalDurationP.value()
    return normalDuration


def phase_converter(phase_timings):
    pass


def AAPILoad():
    return 0


def AAPIInit():
    target_nodes = [3369, 3341, 3370, 3344, 3329]

    catalog = model.getCatalog()
    for nodeid in target_nodes:
        node = catalog.find(nodeid)
        in_edges = node.getEntranceSections()
        for edge in in_edges:
            edge_detector_dict[edge.getId()] = {"stopbar_ids": [],
                                                "advanced_ids": []}

    for i in range(aapi.AKIDetGetNumberDetectors()):
        detector = aapi.AKIDetGetPropertiesDetector(i)
        if detector.IdSection in edge_detector_dict.keys():
            type_map = edge_detector_dict[detector.IdSection]
            edge_aimsun = model.getCatalog().find(detector.IdSection)
            if (edge_aimsun.length2D() - detector.FinalPosition) < 5:
                type_map["stopbar_ids"].append(detector.Id)
            else:
                type_map["advanced_ids"].append(detector.Id)

    node_id = 3344
    cplanType = model.getType("GKControlPlan")

    if aapi.ECIGetNumberofControls(node_id) == 1:
        name = aapi.AKIConvertToAsciiString(aapi.ECIGetNameofControl(node_id, 0), True, aapi.boolp())
        node = catalog.find(node_id)
        cplan_ids = catalog.getObjectsByType(cplanType)
        for cid in cplan_ids:
            cplan = catalog.find(cid)
            if 'AR 5083 - Zone 132 - P2'!=cplan.getName():
                continue
            cjunction = cplan.getControlJunction(node)

            if cjunction is not None:
                # print(cjunction.getOffset(), cplan.getName())
                num_rings = cjunction.getNbRings()
                phase_times = {i+1: OrderedDict() for i in range(num_rings)}
                timing_map = {}
                for phase in cjunction.getPhases():
                    if len(phase.getSignals())>0:
                        ring_id = phase.getIdRing()
                        start_time = phase.getFrom()
                        phase_times[ring_id][start_time] = [i.name for i in phase.getSignals()]
                        timing_map[start_time] = ring_id
                        # print(phase.getFrom(), phase.getIdRing(), [i.name for i in phase.getSignals()])
                print(phase_times)



                # sorted(mydict.items(), key=lambda item: item[1]
    else:
        print("WTH")
    return 0


def AAPIManage(time, timeSta, timeTrans, acycle):
    # a = aapi.doublep()
    # b = aapi.doublep()
    # c = aapi.doublep()
    # aapi.ECIGetDurationsPhase(3344, 13, 3600*7.5, a,b,c)
    # print(a.value(), b.value(), c.value())

    node_id = 3344
    curr_phase = get_current_phase(node_id)
    num_phases = get_num_phases(node_id)

    offset = -20
    if time == 60:
        print(curr_phase)
        curr_time_phase = time-aapi.ECIGetStartingTimePhaseInRing(node_id, 0)
        for i, phase in enumerate(curr_phase):
            target_phase = phase
            targ_duration = get_duration_phase(node_id, phase, timeSta)
            remaining_time = targ_duration - curr_time_phase + offset
            if remaining_time < 0:
                target_phase = phase - 1
                targ_duration = get_duration_phase(node_id, target_phase, timeSta)
                while remaining_time + targ_duration < 0:
                    remaining_time += targ_duration
                    target_phase -= 1  # ensure this wraps around ring
            aapi.ECIChangeDirectPhase(node_id, target_phase, timeSta, time,
                                      acycle, targ_duration - remaining_time)

    # print(timeSta, a.value())
    return 0


def AAPIPostManage(time, timeSta, timeTrans, acycle):
    # print(aapi.ECIGetNumberPhases(3344))
    # print(aapi.ECIGetOffset(3344))  # AR 5083 - Zone 132 - P2, returns 0??
    # print([aapi.ECIGetCurrentPhaseInRing(3344, i) for i in range(2)])
    # print(aapi.ECIGetCurrentStateofMeteringById(3344, 1))
    # print(aapi.ECIGetNumberSem(3344))

    node_id = 3344

        # aapi.ECIChangeTimingPhase(3344, 14, 0, timeSta)
        # aapi.ECIChangeTimingPhase(3344, 13, 0, timeSta)
    if time == 60.8:
        curr_phase = get_current_phase(node_id)
        print(curr_phase)
        print(aapi.ECIGetCurrentTimeInCycle(node_id, 0))


    if time % 300 == 0:
        # read sensors:
        section_id = 568
        # section_id = 461


        # anyNonAsciiChar = aapi.boolp()
        # s = aapi.AKIConvertToAsciiString(aapi.ECIGetNameofControl(3344, 0), 
        #                                  True,
        #                                  anyNonAsciiChar)
        # print(s)
        # print("occupancy:", get_section_occupancy(section_id))
        # print("flow:", get_section_flow(section_id))
    return 0


def AAPIFinish():
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