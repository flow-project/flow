import AAPI as aapi
import csv
import PyANGKernel as gk
import PyANGConsole as cs
from datetime import datetime

model = gk.GKSystem.getSystem().getActiveModel()
# global edge_detector_dict
# edge_detector_dict = {}


now = datetime.now()
westbound_section = [506, 563, 24660, 568, 462]
eastbound_section = [338, 400, 461, 24650, 450]
sections = [22208, 568, 22211, 400]
node_id = 3344

interval = 3*60

replication_name = aapi.ANGConnGetReplicationId()
replication = model.getCatalog().find(8050315)
current_time = now.strftime('%d-%m-%Y-%H-%M:%S')

with open('{}.csv'.format(replication_name), 'w') as csvFile:
    data = []
    fieldnames = ['section_id', 'time', 'delay_time', 'travel_time']
    csv_writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
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

    print('SUM QUEUE {} : {}'.format(node_id, total_queue))


def AAPILoad():
    return 0


def AAPIInit():
    return 0


def AAPIManage(time, timeSta, timeTrans, acycle):
    # print( "AAPIManage" )
    return 0


def AAPIPostManage(time, timeSta, timeTrans, acycle):
    # print( "AAPIPostManage" )
    """if time % (15*60) == 0:
        for section_id in sections:
            estad = aapi.AKIEstGetGlobalStatisticsSection(section_id, 0)
            if (estad.report == 0):
                dta = estad.DTa
                tta = estad.TTa
                time = time
                #print('dt: {:.4f}, tt: {:.4f}'.format(estad.DTa, estad.TTa))
                # print('\n Mean Queue: \t {}'.format(estad.))

                if replication_name == 8050297:
                    with open('{}.csv'.format(replication_name), 'a') as csvFile:
                        csv_writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
                        csv_writer.writerow({'section_id': '{}'.format(section_id), 'time': '{}'.format(time), 'delay_time': '{: .4f}'.format(dta),
                                             'travel_time': '{:.4f}'.format(tta)})

                elif replication_name == 8050315:
                    with open('{}.csv'.format(replication_name), 'a') as csvFile:
                        csv_writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
                        csv_writer.writerow({'section_id': '{}'.format(section_id), 'time': '{}'.format(time), 'delay_time': '{: .4f}'.format(dta),
                                             'travel_time': '{:.4f}'.format(tta)})

                if replication_name == 8050322:
                    with open('{}.csv'.format(replication_name), 'a') as csvFile:
                        csv_writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
                        csv_writer.writerow({'section_id': '{}'.format(section_id), 'time': '{}'.format(time), 'delay_time': '{: .4f}'.format(dta),
                                             'travel_time': '{:.4f}'.format(tta)})"""

    # console = cs.ANGConsole()
    if time == interval:
        print('yey')
        aapi.ANGSetSimulationOrder(1, interval)
        # aapi.ANGSetSimulationOrder(2, 0)
        # replication = model.getCatalog().find(replication_name)
        gk.GKSystem.getSystem().executeAction("execute", replication, [], "")
        # console.close()
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
