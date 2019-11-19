import AAPI as aapi
import PyANGKernel as gk
from collections import OrderedDict
import numpy as np

model = gk.GKSystem.getSystem().getActiveModel()
global edge_detector_dict
edge_detector_dict = {}


def get_detector_flow_and_occupancy(detector_id):
    flow = max(aapi.AKIDetGetCounterAggregatedbyId(detector_id, 0), 0)
    occupancy = max(aapi.AKIDetGetTimeOccupedAggregatedbyId(detector_id, 0), 0)/100
    return flow, occupancy


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


def change_offset(node_id, offset, time, timeSta, acycle):
    curr_phase = get_current_phase(node_id)
    ring_phases = np.cumsum(get_num_phases(node_id))

    elapsed_time = time-aapi.ECIGetStartingTimePhaseInRing(node_id, 0)
    for i, phase in enumerate(curr_phase):
        target_phase = phase
        phase_time = get_duration_phase(node_id, phase, timeSta)
        remaining_time = (phase_time - elapsed_time) + offset
        while remaining_time < 0:
            target_phase += 1
            if target_phase > ring_phases[i]:
                target_phase -= ring_phases[0]

            phase_time = get_duration_phase(node_id, target_phase, timeSta)
            remaining_time += phase_time
        aapi.ECIChangeDirectPhase(node_id, target_phase, timeSta, time,
                                  acycle, phase_time - remaining_time)


def phase_converter(phase_timings):
    pass


def get_combined_ring(start_times, phase_times):
    combined_ring = {time: [] for time in sorted(start_times)}
    for time in combined_ring.keys():
        for ring in phase_times.keys():
            last_phase_time = None
            for phase_time in phase_times[ring].keys():
                if phase_time - time <= 0:
                    last_phase_time = phase_time
                else:
                    break

            for phase in phase_times[ring][last_phase_time]:
                combined_ring[time].append(phase)

    return combined_ring


def get_link_measures(target_nodes):
    model = gk.GKSystem.getSystem().getActiveModel()
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
            edge_aimsun = catalog.find(detector.IdSection)
            if (edge_aimsun.length2D() - detector.FinalPosition) < 5:
                kind = "stopbar_ids"
            else:
                kind = "advanced_ids"
            detector_obj = catalog.find(detector.Id)
            try:
                # only those with numerical exernalIds are real
                int(detector_obj.getExternalId())
                type_map[kind].append(detector.Id)
            except ValueError:
                pass

    return edge_detector_dict


def AAPILoad():
    # model.setActiveExperimentId(8050312)
    # print(model.getActiveExperimentId(), "experiment id")

    # model.setActiveReplicationId(8050315)
    # print(model.getActiveReplicationId(), "replication id")
    return 0


def maxband():
    target_nodes = {3369: "AR 5083 - Zone 132 - P2",
                    3341: "AR 5083 - Zone 132 - P2",
                    3370: "AR 5083 - Zone 132 - P2",
                    3344: "AR 5083 - Zone 132 - P2",
                    3329: "AR 5083 - Zone 132 - P2"}

    b = []
    b_bar = []
    w = []
    w_bar = []
    delta = []
    r = []
    r_bar = []
    cycle_length = 120

    for node_id, cplan_name in target_nodes.items():
        cplanType = model.getType("GKControlPlan")

        catalog = model.getCatalog()
        if aapi.ECIGetNumberofControls(node_id) == 1:
            node = catalog.find(node_id)
            cplan_ids = catalog.getObjectsByType(cplanType)
            for cid in cplan_ids:
                # Find current active control plan
                cplan = catalog.find(cid)
                if cplan_name != cplan.getName():  # currently specific to a node
                    continue
                cjunction = cplan.getControlJunction(node)

                if cjunction:
                    num_rings = cjunction.getNbRings()
                    phase_times = {i+1: OrderedDict() for i in range(num_rings)}
                    timing_map = {}
                    # start_times = set()
                    phase_ids = [3, 11]  # TBD
                    # for phase_id in phase_ids:
                    phase = cjunction.getPhaseByPos(phase_ids[0] - 1)
                    phase_bar = cjunction.getPhaseByPos(phase_ids[1] - 1)

                    # if (len(phase.getSignals()) > 0) and (phase.getIdRing() == 1):  # might be safe to remove
                    # force phase
                    # ring_id = phase.getIdRing()
                    start_time = phase.getFrom()
                    duration_green = phase.getDuration()
                    duration_red = cycle_length - duration_green
                    r.append(duration_red)
                    r_midway = (start_time + duration_green + duration_red/2) % cycle_length

                    # force phase 11
                    start_time_bar = phase_bar.getFrom()
                    duration_green_bar = phase_bar.getDuration()
                    duration_red_bar = cycle_length - duration_green_bar
                    r_bar.append(duration_red_bar)
                    r_bar_midway = (start_time_bar + duration_green_bar + duration_red_bar/2) % cycle_length

                    delta.append(r_midway - r_bar_midway)

                    # start_times.add(start_time)
                    phase_times[ring_id][start_time] = [i.name for i in phase.getSignals()]
                    timing_map[start_time] = ring_id

                    # combined_ring = get_combined_ring(start_times, phase_times)


def AAPIInit():
    target_nodes = [3369, 3341, 3370, 3344, 3329]

    edge_detector_dict = get_link_measures(target_nodes)
    print(edge_detector_dict)

    node_id = 3344
    cplanType = model.getType("GKControlPlan")

    # model = GKSystem.getSystem().getActiveModel()
    catalog = model.getCatalog()
    if aapi.ECIGetNumberofControls(node_id) == 1:
        name = aapi.AKIConvertToAsciiString(aapi.ECIGetNameofControl(node_id, 0), True, aapi.boolp())
        node = catalog.find(node_id)
        cplan_ids = catalog.getObjectsByType(cplanType)
        for cid in cplan_ids:
            cplan = catalog.find(cid)
            if 'AR 5083 - Zone 132 - P2' != cplan.getName():
                continue
            cjunction = cplan.getControlJunction(node)

            if cjunction is not None:
                # print(cjunction.getOffset(), cplan.getName())
                num_rings = cjunction.getNbRings()
                phase_times = {i+1: OrderedDict() for i in range(num_rings)}
                timing_map = {}
                start_times = set()
                for phase in cjunction.getPhases():
                    if len(phase.getSignals()) > 0:
                        ring_id = phase.getIdRing()
                        start_time = phase.getFrom()
                        start_times.add(start_time)
                        phase_times[ring_id][start_time] = [i.name for i in phase.getSignals()]
                        timing_map[start_time] = ring_id
                        # print(phase.getFrom(), phase.getIdRing(), [i.name for i in phase.getSignals()])
                # print(phase_times)

                combined_ring = get_combined_ring(start_times, phase_times)

                # sorted(mydict.items(), key=lambda item: item[1]
    else:
        print("WTH")

    return 0


global q
q = 0


def AAPIManage(time, timeSta, timeTrans, acycle):
    node_id = 3344
    curr_phase = get_current_phase(node_id)
    ring_phases = np.cumsum(get_num_phases(node_id))
    global q
    if time > 15:
        print('stopping', time)
        aapi.ANGSetSimulationOrder(1, 0)
        print('sim canceled')

        # aapi.ANGSetSimulationOrder(0, 0)
        replication_id = model.getActiveReplicationId()
        replication = model.getCatalog().find(replication_id)
        gk.GKSystem.getSystem().executeAction("play", replication, [], "")
        print('sim played')

    # print(aapi.ECIGetOffset(3370), 'offset')
    offset = -70
    if time % 300 == 0:
        # replications = model.getCatalog().getObjectsByType(model.getType("GKReplication"))
        # for replication in replications.values():
        #     print(replication.getDBId())
        catalog = model.getCatalog()
        section = catalog.find(461)
        num_lanes = section.getNbLanesAtPos(section.length2D())
        queue = sum(aapi.AKIEstGetCurrentStatisticsSectionLane(461, i, 0).LongQueueAvg for i in range(num_lanes))

        # queue = aapi.AKIEstGetParcialStatisticsSection(461, timeSta, 0).LongQueueAvg
        # print(queue-q, num_lanes)
        q = queue
        7674941, 7674942
    catalog = model.getCatalog()
    replication = catalog.find(8050330)
    if time % 60 == 0:

        # print(curr_phase)
        change_offset(node_id, offset, time, timeSta, acycle)

    # print(timeSta, a.value())
    return 0


def AAPIPostManage(time, timeSta, timeTrans, acycle):
    node_id = 3344
    if time == 60.8:
        curr_phase = get_current_phase(node_id)
        # print(curr_phase)
    # if aapi.AKIEstIsNewStatisticsAvailable():
    #     print(time)
    return 0


def AAPIFinish():
    return 0


def AAPIUnLoad():
    print('unload')
    replication_id = model.getActiveReplicationId()
    replication = model.getCatalog().find(replication_id)
    gk.GKSystem.getSystem().executeAction("play", replication, [], "")
    print('sim played')
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
