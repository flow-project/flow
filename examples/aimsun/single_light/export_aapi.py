import AAPI as aapi
import sys
import csv
import PyANGKernel as gk
import PyANGConsole as cs
from datetime import datetime
sys.path.append('/home/cjrsantos/anaconda3/envs/aimsun_flow/lib/python2.7/site-packages')
import numpy as np

model = gk.GKSystem.getSystem().getActiveModel()
# global edge_detector_dict
# edge_detector_dict = {}


now = datetime.now()
westbound_section = [506, 563, 24660, 568, 462]
eastbound_section = [338, 400, 461, 24650, 450]
sections = [22208, 568, 22211, 400]
node_id = 3344

interval = 15*60
#seed = np.random.randint(2e9)

replication_name = aapi.ANGConnGetReplicationId()
replication = model.getCatalog().find(8050315)
current_time = now.strftime('%H-%M:%S')

with open('{}.csv'.format(replication_name), 'a') as csvFile:
    data = []
    fieldnames = ['time', 'delay_time', '1', '3', '5', '7', '9', '11', '13', '15', 'cycle', 'barrier']
    csv_writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
    csv_writer.writerow({'time': current_time})
    csv_writer.writeheader()


def get_delay_time(section_id):
    west = []
    east = []
    for section_id in westbound_section:
        estad_w = aapi.AKIEstGetGlobalStatisticsSection(section_id, 0)
        if estad_w.report == 0:
            print('Delay time: {} - {}'.format(section_id, estad_w.DTa))
        west.append(estad_w.DTa)

    for section_id in eastbound_section:
        estad_e = aapi.AKIEstGetGlobalStatisticsSection(section_id, 0)
        if estad_e.report == 0:
            print('Delay time: {} - {}'.format(section_id, estad_e.DTa))
        east.append(estad_e.DTa)

    west_ave = sum(west)/len(west)
    east_ave = sum(east)/len(east)

    print("Average Delay Time: WestBound {}".format(west_ave))
    print("Average Delay Time: EastBound {}".format(east_ave))


def sum_queue(section_id):
    catalog = model.getCatalog()
    node = catalog.find(node_id)
    in_edges = node.getEntranceSections()

    section_list = [edge.getId() for edge in in_edges]

    for section_id in section_list:
        section = catalog.find(section_id)
        num_lanes = section.getNbLanesAtPos(section.length2D())
        queue = sum(aapi.AKIEstGetCurrentStatisticsSectionLane(
            section_id, i, 0).LongQueueAvg for i in range(num_lanes))

        queue = queue * 5 / section.length2D()

    print('SUM QUEUE {} : {}'.format(node_id))

def set_replication_seed(seed):
    replications = model.getCatalog().getObjectsByType(model.getType("GKReplication"))
    for replication in replications.values():
        replication.setRandomSeed(seed)

def get_ttadta(section_id, timeSta):
    # print( "AAPIPostManage" )
    if time % (15*60) == 0:
        for section_id in sections:
            estad = aapi.AKIEstGetParcialStatisticsSection(section_id, timeSta, 0)
            if (estad.report == 0):
                dta = estad.DTa
                tta = estad.TTa
                time = time
                # print('dt: {:.4f}, tt: {:.4f}'.format(estad.DTa, estad.TTa))
                # print('\n Mean Queue: \t {}'.format(estad.))


def get_control_ids(node_id):
    control_id = aapi.ECIGetNumberCurrentControl(node_id)
    num_rings = aapi.ECIGetNbRingsJunction(control_id, node_id)

    return control_id, num_rings


def get_cycle_length(node_id, control_id):  # cj
    # Format is set current control plan
    control_cycle = aapi.ECIGetControlCycleofJunction(control_id, node_id)
    return control_cycle


def get_phases(ring_id):
    phases = []
    num_phases = aapi.ECIGetNumberPhasesInRing(node_id, ring_id)
    for phase in range(1, num_phases*2+1):
        phases.append(phase)
    return phases

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

def get_phase_duration_list(node_id, timeSta):
    control_id, num_rings = get_control_ids(node_id)
    cycle = get_cycle_length(node_id, control_id)
    phase_list = get_phases(0)
    dur_list = []
    for phase in phase_list:
        if aapi.ECIIsAnInterPhase(node_id, phase, timeSta) == 1:
            continue
        else:
            dur, _, _ = get_duration_phase(3344, phase, timeSta)
            idur = int(dur)
            dur_list.append(idur)
    return dur_list, cycle


def AAPILoad():
    return 0


def AAPIInit():
    #set_replication_seed(seed)
    #print(seed)
    return 0


def AAPIManage(time, timeSta, timeTrans, acycle):
    # print( "AAPIManage" )
    return 0


def AAPIPostManage(time, timeSta, timeTrans, acycle):
    # print( "AAPIPostManage" )
    if time % (15*60) == 0:
        time = time
        timeSta = timeSta
        ave_app_delay = aapi.AKIEstGetPartialStatisticsNodeApproachDelay(node_id)
        dur_list, cycle= get_phase_duration_list(node_id, timeSta)
        barrier = (sum(dur_list[0:2]), sum(dur_list[2:4]))
           # print('dt: {:.4f}, tt: {:.4f}'.format(estad.DTa, estad.TTa))
           # print('\n Mean Queue: \t {}'.format(estad.))

        if replication_name == 8050297:
            with open('{}.csv'.format(replication_name), 'a') as csvFile:
                csv_writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
                csv_writer.writerow({'time': '{}'.format(time), 'delay_time': '{}'.format(ave_app_delay), 
                    '1': '{}'.format(dur_list[0]), '3': '{}'.format(dur_list[1]), '5': '{}'.format(dur_list[2]), 
                    '7': '{}'.format(dur_list[3]), '9': '{}'.format(dur_list[4]), '11': '{}'.format(dur_list[5]), 
                    '13': '{}'.format(dur_list[6]), '15': '{}'.format(dur_list[7]),
                    'cycle': '{}'.format(cycle), 'barrier': '{}'.format(barrier)})

        elif replication_name == 8050315:
            with open('{}.csv'.format(replication_name), 'a') as csvFile:
                csv_writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
                csv_writer.writerow({'time': '{}'.format(time), 'delay_time': '{}'.format(ave_app_delay), 
                    '1': '{}'.format(dur_list[0]), '3': '{}'.format(dur_list[1]), '5': '{}'.format(dur_list[2]), 
                    '7': '{}'.format(dur_list[3]), '9': '{}'.format(dur_list[4]), '11': '{}'.format(dur_list[5]), 
                    '13': '{}'.format(dur_list[6]), '15': '{}'.format(dur_list[7]),
                    'cycle': '{}'.format(cycle), 'barrier': '{}'.format(barrier)})

        if replication_name == 8050322:
            with open('{}.csv'.format(replication_name), 'a') as csvFile:
                csv_writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
                csv_writer.writerow({'time': '{}'.format(time), 'delay_time': '{}'.format(ave_app_delay), 
                    '1': '{}'.format(dur_list[0]), '3': '{}'.format(dur_list[1]), '5': '{}'.format(dur_list[2]), 
                    '7': '{}'.format(dur_list[3]), '9': '{}'.format(dur_list[4]), '11': '{}'.format(dur_list[5]), 
                    '13': '{}'.format(dur_list[6]), '15': '{}'.format(dur_list[7]),
                    'cycle': '{}'.format(cycle), 'barrier': '{}'.format(barrier)})

    """# console = cs.ANGConsole()
    if time == interval:
        print('yey')
        aapi.ANGSetSimulationOrder(1, interval)
        # aapi.ANGSetSimulationOrder(2, 0)
        # replication = model.getCatalog().find(replication_name)
        gk.GKSystem.getSystem().executeAction("execute", replication, [], "")
        # console.close()"""
    return 0


def AAPIFinish():
    # print("AAPIFinish")
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
