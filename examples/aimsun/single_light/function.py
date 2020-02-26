new_function.py
def get_ids(node_id):
    #returns number of rings and control_id
    control_id = aapi.ECIGetNumberCurrentControl(node_id)
    num_rings = aapi.ECIGetNbRingsJunction(control_id, node_id)

    return control_id, num_rings

    # returns control_id = int, num_rings = int

def get_green_phases(node_id, ring_id, timeSta):
    num_phases = aapi.ECIGetNumberPhasesInRing(node_id, ring_id)
    if ring_id > 0:
        a = num_phases + 1
        num_phases = num_phases*2
            
    return [phase for phase in range(a, num_phases+1)]

    # returns phases = [1 2 3 4]

def __init__(self):
    self.phases = []
    for node_id in target_nodes:
        self.control_id, self.num_rings = self.k.traffic_light.get_ids(node_id)

    for ring_id in range(num_rings):
        phase_list = self.k.traffic_light.get_phases(node_id, ring_id)
        self.phases.append(phase_list)

    # phases = [[1 3 5 7], [9 11 13 15]]

##### RL_ACTION #####
def change_phase_duration(node_id, phase, duration, time, timeSta, acycle):
    aapi.ECIChangeTimingPhase(node_id, phase, duration, timeSta)

@property
def action_space(self):
    """See class definition."""
    return Tuple(4 * (Discrete(120, ),)) #*fixed number (number of green phases per ring) [hardcoded]

def _apply_rl_actions(self, rl_actions):
    if self.ignore_policy:
        print('self.ignore_policy is True')
        return
    actions = np.array(rl_actions).flatten()
    for phase_list in self.phases:
        for phase, action in zip(phase_list, actions):
            if action:
                self.k.traffic_light.change_phase_duration(phase, duration)
    self.current_phase_timings = actions










### What do i need: ###
    control_id, ring_id

def get_current_ids(node_id):
    phases = []
    control_id = aapi.ECIGetNumberCurrentControl(node_id) #Get control index
    num_rings = aapi.ECIGetNbRingsJunction(control_id,node_id) #Get Number of rings
    for ring_id in range(1,num_rings*2):
        num_phases = aapi.ECIGetNumberPhasesInRing(node_id, ring_id)
    for phase in range(num_phases):
        phases.append(phase)

    return control_id, num_rings, phases

def change_phase_timing(node_id, duration, cycle, time, timeSta, acycle):
    control_id = aapi.ECIGetNumberCurrentControl(node_id) #Get control index
    num_rings = aapi.ECIGetNbRingsJunction(control_id,node_id) #Get Number of rings
    for ring_id in range(num_rings):
        curr_phase = aapi.ECIGetCurrentPhaseInRing(node_id,ring_id)
        num_phases = aapi.ECIGetNumberPhasesInRing(node_id, ring_id)
        if curr_phase == num_phases: ##can be changed in interval time? Should I disregard interval time? Should i just per 15mins? 
            total_green = int(get_total_green(node_id, ring_id, timeSta))
            #####
            a = 1
            if ring_id > 0:
                a = num_phases + 1
                num_phases = num_phases*2
            for phase in range(a,num_phases+1):
                if aapi.ECIIsAnInterPhase(node_id, phase, timeSta) == 1:
                    continue
                else:
                    #### can include that duration sum must be equal to total green
                    #### max green and min green parameters
                    aapi.ECIChangeTimingPhase(node_id, phase, duration, timeSta)
            duration = get_total_green(node_id, ring_id, timeSta)
            print('Ring ID: {}, Duration:{}'.format(ring_id,duration))
    return 0

def get_green_phases(node_id, ring_id, timeSta):
    green_phases = []
    a = 1
    num_phases = aapi.ECIGetNumberPhasesInRing(node_id, ring_id)
    if ring_id > 0:
        a = num_phases + 1
        num_phases = num_phases*2
    for phase in range(a,num_phases+1):
        if aapi.ECIIsAnInterPhase(node_id, phase, timeSta) == 1:
            continue
        else:
            green_phases.append(phase)
    return green_phases

    # returns phases = [1 2 3 4]



#####First Trial####

import AAPI as aapi
import numpy as np

def change_phase_timing(node_id, duration, cycle, time, timeSta, acycle):
    control_id = aapi.ECIGetNumberCurrentControl(node_id) #Get control index
    num_rings = aapi.ECIGetNbRingsJunction(control_id,node_id) #Get Number of rings
    for ring_id in range(num_rings):
        curr_phase = aapi.ECIGetCurrentPhaseInRing(node_id,ring_id)
        num_phases = aapi.ECIGetNumberPhasesInRing(node_id, ring_id)
        if curr_phase == num_phases: ##can be changed in interval time? Should I disregard interval time? Should i just per 15mins? 
            total_green = int(get_total_green(node_id, ring_id, timeSta))
            #####
            a = 1
            if ring_id > 0:
                a = num_phases + 1
                num_phases = num_phases*2
            for phase in range(a,num_phases+1):
                if aapi.ECIIsAnInterPhase(node_id, phase, timeSta) == 1:
                    continue
                else:
                	#### can include that duration sum must be equal to total green
                	#### max green and min green parameters
                    aapi.ECIChangeTimingPhase(node_id, phase, duration, timeSta)
            duration = get_total_green(node_id, ring_id, timeSta)
            print('Ring ID: {}, Duration:{}'.format(ring_id,duration))
    return 0

def get_total_green(node_id, ring_id, timeSta):
    ## Format is set current control plan 
    sum_interphase = 0
    control_id = aapi.ECIGetNumberCurrentControl(node_id)
    control_cycle = aapi.ECIGetControlCycleofJunction(control_id, node_id)
    num_phases = aapi.ECIGetNumberPhasesInRing(node_id, ring_id)
    for phase in (range(1,num_phases + 1)):
        if aapi.ECIIsAnInterPhase(node_id, phase, timeSta) == 0:
            continue
        else:
            duration = get_duration_phase(node_id, phase, timeSta)
            sum_interphase += duration
 
    total_green = control_cycle - sum_interphase
    return total_green

def change_barrier_timing(node_id,timeSta):
    # For Current Control Plan
    control_id = aapi.ECIGetNumberCurrentControl(node_id) #Get control index
    num_rings = aapi.ECIGetNbRingsJunction(control_id,node_id) #Get Number of rings
    total_green = int(get_total_green(node_id, 0, timeSta))
    for ring_id in range(num_rings):
        num_phases = aapi.ECIGetNumberPhasesInRing(node_id, ring_id)
        a = 1
        if ring_id > 0:
            a = num_phases + 1
            num_phases = num_phases*2
        for phase in range(a,num_phases+1):
            if aapi.ECIIsAnInterPhase(node_id, phase, timeSta) == 1:
                continue
            else:
                aapi.ECIChangeTimingPhase(node_id, phase, duration, timeSta)
    return 0

 
















