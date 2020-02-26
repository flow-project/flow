import AAPI as aapi
import PyANGKernel as gk
import numpy as np

model = gk.GKSystem.getSystem().getActiveModel()
global edge_detector_dict
edge_detector_dict = {}
length_car = 5  # typical car length


def get_intersection_offset(node_id):
    return aapi.ECIGetOffset(node_id)


def get_cumulative_queue_length(section_id):
    catalog = model.getCatalog()
    section = catalog.find(section_id)
    num_lanes = section.getNbLanesAtPos(section.length2D())
    queue = sum(aapi.AKIEstGetCurrentStatisticsSectionLane(section_id, i, 0).LongQueueAvg for i in range(num_lanes))

    return queue*length_car/section.length2D()


def set_replication_seed(seed):
    replications = model.getCatalog().getObjectsByType(model.getType("GKReplication"))
    for replication in replications.values():
        replication.setRandomSeed(seed)


def set_statistical_interval(hour, minute, sec):
    time_duration = gk.GKTimeDuration(hour, minute, sec)
    scenarios = model.getCatalog().getObjectsByType(model.getType("GKScenario"))
    for scenario in scenarios.values():
        input_data = scenario.getInputData()
        input_data.setStatisticalInterval(time_duration)


def set_detection_interval(hour, minute, sec):
    time_duration = gk.GKTimeDuration(hour, minute, sec)
    scenarios = model.getCatalog().getObjectsByType(model.getType("GKScenario"))
    for scenario in scenarios.values():
        input_data = scenario.getInputData()
        input_data.setDetectionInterval(time_duration)


def get_detector_count_and_occupancy(detector_id):
    flow = max(aapi.AKIDetGetCounterAggregatedbyId(detector_id, 0), 0)
    occupancy = max(aapi.AKIDetGetTimeOccupedAggregatedbyId(detector_id, 0), 0)/100
    return flow, occupancy


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
    maxDuration = maxDurationP.value()
    minDuration = minDurationP.value()

    return normalDuration, maxDuration, minDuration


def change_offset(node_id, offset, time, timeSta, acycle):
    curr_phase = get_current_phase(node_id)
    ring_phases = np.cumsum(get_num_phases(node_id))

    elapsed_time = time-aapi.ECIGetStartingTimePhaseInRing(node_id, 0)
    for i, phase in enumerate(curr_phase):
        target_phase = phase
        phase_time, _, _ = get_duration_phase(node_id, phase, timeSta)
        remaining_time = (phase_time - elapsed_time) + offset
        while remaining_time < 0:
            target_phase += 1
            if target_phase > ring_phases[i]:
                target_phase -= ring_phases[0]

            phase_time, _, _ = get_duration_phase(node_id, target_phase, timeSta)
            remaining_time += phase_time
        aapi.ECIChangeDirectPhase(node_id, target_phase, timeSta, time, acycle, phase_time - remaining_time)


def set_max_green(control_id, node_id, phase_id, new_duration):
    aapi.ECISetActuatedParamsMaxGreen(control_id, node_id, phase_id, new_duration)


def get_total_green(node_id, timeSta):  # cj
    # Format is set current control plan
    ring_id = 0
    sum_interphase = 0
    control_id = aapi.ECIGetNumberCurrentControl(node_id)
    control_cycle = aapi.ECIGetControlCycleofJunction(control_id, node_id)
    num_phases = aapi.ECIGetNumberPhasesInRing(node_id, ring_id)
    for phase in (range(1, num_phases + 1)):
        if aapi.ECIIsAnInterPhase(node_id, phase, timeSta) == 0:
            continue
        else:
            duration, _, _ = get_duration_phase(node_id, phase, timeSta)
            sum_interphase += duration

    total_green = control_cycle - sum_interphase
    return total_green


def get_control_ids(node_id):
    control_id = aapi.ECIGetNumberCurrentControl(node_id)
    num_rings = aapi.ECIGetNbRingsJunction(control_id, node_id)

    return control_id, num_rings


def get_green_phases(node_id, ring_id, timeSta):
    a = 1
    num_phases = aapi.ECIGetNumberPhasesInRing(node_id, ring_id)
    if ring_id > 0:
        a = num_phases + 1
        num_phases = num_phases*2

    return [phase for phase in range(a, num_phases+1) if aapi.ECIIsAnInterPhase(node_id, phase, timeSta) == 0]


def change_phase_duration(node_id, phase, duration, time, timeSta, acycle):
    control_id, _ = get_control_ids(node_id)
    aapi.ECIChangeTimingPhase(node_id, phase, duration, timeSta)
    phase_duration, maxd, mind = get_duration_phase(node_id, phase, timeSta)
    if duration > maxd:
        set_max_green(control_id, node_id, phase, duration)


def phase_converter(phase_timings):
    raise NotImplementedError


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


def get_incoming_edges(node_id):
    catalog = model.getCatalog()
    node = catalog.find(node_id)
    in_edges = node.getEntranceSections()

    return [edge.getId() for edge in in_edges]


def get_detector_ids(edge_id):
    catalog = model.getCatalog()
    detector_list = {"stopbar": [], "advanced": []}
    for i in range(aapi.AKIDetGetNumberDetectors()):
        detector = aapi.AKIDetGetPropertiesDetector(i)
        if detector.IdSection == edge_id:
            edge_aimsun = catalog.find(detector.IdSection)

            if (edge_aimsun.length2D() - detector.FinalPosition) < 5:
                kind = "stopbar"
            else:
                kind = "advanced"

            detector_obj = catalog.find(detector.Id)
            try:
                # only those with numerical exernalIds are real
                int(detector_obj.getExternalId())
                detector_list[kind].append(detector.Id)
            except ValueError:
                pass
    return detector_list
