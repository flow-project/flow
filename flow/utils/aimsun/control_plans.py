import AAPI as aapi
from PyANGKernel import *
from collections import OrderedDict
import numpy as np

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

    return get_current_phase(node_id)

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
