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





### for the environment
def _apply_rl_actions(self, rl_actions):
    if self.ignore_policy:
        print('self.ignore_policy is True')
        return

    actions = np.array(rl_actions).flatten()
    if sum(actions) == self.total_green:
        continue
    else:
        ##repeaet such that sum == total_green





























	"""control_id = aapi.ECIGetNumberCurrentControl(node_id)
	num_phase = aapi.ECIGetNbPhasesofJunction(control_id,node_id)
	ring_id = 1
	for phase in range(1,num_phase+1):
		if aapi.ECIIsAnInterPhase(node_id, phase, timeSta) == 1:
			continue
		else:
			current_duration = get_duration_phase(node_id, phase, timeSta)
			aapi.ECIChangeTimingPhase(node_id, phase, duration, timeSta)
			##where in sum_duration_phases must total to cycle
		duration, cycle = get_sum_duration_and_cycle(node_id, ring_id, timeSta)"""