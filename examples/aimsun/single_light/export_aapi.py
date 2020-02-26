import AAPI as aapi
import csv

# model = gk.GKSystem.getSystem().getActiveModel()
# global edge_detector_dict
# edge_detector_dict = {}

westbound_section = [506, 563, 24660, 568, 462]
eastbound_section = [338, 400, 461, 24650, 450]


def get_delay_time(section_id):
    pass

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
    if time % 900 == 0:
        control_id, num_rings, phases = get_current_ids(3344)
        print(control_id)
        print(num_rings)
        print(phases)

    return 0


def AAPIFinish():
    print("AAPIFinish")
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
