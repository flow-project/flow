"""Script containing objects used to store vehicle information in Aimsun."""


class InfVeh(object):
    """Dynamics (tracking) information for vehicles in Aimsun.

    Attributes
    ----------
    CurrentPos : float
        Position inside the section. The distance (metres or feet, depending on
        the units defined in the network) from the beginning of the section or
        position inside the junction given as the distance from the entrance to
        the junction
    distance2End : float
        Distance to end of the section (metres or feet, depending on the units
        defined in the network) when the vehicle is located in a section or the
        distance to the end of the turn when the vehicle is in a junction
    xCurrentPos : float
        x coordinates of the middle point of the front bumper of the vehicle
    yCurrentPos : float
        y coordinates of the middle point of the front bumper of the vehicle
    zCurrentPos : float
        z coordinates of the middle point of the front bumper of the vehicle
    xCurrentPosBack : float
        x coordinates of the middle point of the rear bumper of the vehicle
    yCurrentPosBack : float
        y coordinates of the middle point of the rear bumper of the vehicle
    zCurrentPosBack : float
        z coordinates of the middle point of the rear bumper of the vehicle
    CurrentSpeed : float
        Current speed (in km/h or mph, depending on the units defined in the
        network)
    TotalDistance : float
        Total distance travelled (metres or feet)
    SectionEntranceT : float
        The absolute entrance time of the vehicle into the current section
    CurrentStopTime : float
        The current stop time
    stopped : float
        True if the vehicle remains stopped
    idSection : int
        The section identifier
    segment : int
        Segment number of the section where the vehicle is located (from 0 to
        n-1)
    numberLane : int
        Lane number in the segment (from 1, the rightmost lane, to N, the
        leftmost lane)
    idJunction : int
        The junction identifier
    idSectionFrom : int
        Origin section identifier when the vehicle is in a node
    idLaneFrom : int
        Origin sections lane where the vehicle enters the junction from. 1
        being the rightmost lane and N the leftmost lane, being N the number of
        lanes in the origin section
    idSectionTo : int
        Destination section identifier when the vehicle is in a node
    idLaneTo : int
        Destination sections lane where the vehicle exits the junction to. 1
        being the rightmost lane and N the leftmost lane, being N the number of
        lanes in the destination section
    """

    def __init__(self):
        """Instantiate InfVeh."""
        self.CurrentPos = None
        self.distance2End = None
        self.xCurrentPos = None
        self.yCurrentPos = None
        self.zCurrentPos = None
        self.xCurrentPosBack = None
        self.yCurrentPosBack = None
        self.zCurrentPosBack = None
        self.CurrentSpeed = None
        self.TotalDistance = None
        self.SectionEntranceT = None
        self.CurrentStopTime = None
        self.stopped = None

        # Information in Vehicle when it is in a section
        self.idSection = None
        self.segment = None
        self.numberLane = None

        # Information in Vehicle when it is in a node
        self.idJunction = None
        self.idSectionFrom = None
        self.idLaneFrom = None
        self.idSectionTo = None
        self.idLaneTo = None


class StaticInfVeh(object):
    """Static information for vehicles in Aimsun.

    Attributes
    ----------
    report : int
        0, OK, else error code
    idVeh : int
        Vehicle identifier
    type : int
        Vehicle type (car, bus, truck, etc.)
    length : float
        Vehicle length (m or feet, depending on the units defined in the
        network)
    width : float
        Vehicle width (m or feet, depending on the units defined in the
        network)
    maxDesiredSpeed : float
        Maximum desired speed of the vehicle (km/h or mph, depending on the
        units defined in the network)
    maxAcceleration : float
        Maximum acceleration of the vehicle (m/s2 or ft/ s2, depending on
        the units defined in the network)
    normalDeceleration : float
        Maximum deceleration of the vehicle that can apply under normal
        conditions (m/s2 or ft/ s2, depending the units defined in the
        network)
    maxDeceleration : float
        Maximum deceleration of the vehicle that can apply under special
        conditions (m/s2 or ft/ s2, depending the units defined in the
        network)
    speedAcceptance : float
        Degree of acceptance of the speed limits
    minDistanceVeh : float
        Distance that the vehicle keeps between itself and the preceding
        vehicle (metres or feet, depending on the units defined in the
        network)
    giveWayTime : float
        Time after which the vehicle becomes more aggressive in give-way
        situations (seconds)
    guidanceAcceptance : float
        Level of compliance of the vehicle to guidance indications
    enrouted : int
        0 means vehicle will not change path enroute, 1 means vehicle will
        change path enroute depending on the percentage of enrouted
        vehicles defined
    equipped : int
        0 means vehicle not equipped, 1 means vehicle equipped
    tracked : int
        0 means vehicle not tracked, 1 means vehicle tracked
    keepfastLane : bool
        True means the vehicle keeps fast lane during overtaking
    headwayMin : float
        Minimum headway to the leader
    sensitivityFactor : float
        Estimation of the acceleration of the leader
    reactionTime : float
        Reaction time of the vehicle
    reactionTimeAtStop : float
        Reaction time at stop of the vehicle
    reactionTimeAtTrafficLight : float
        Reaction time of the vehicle when stopped the first one of the
        queue in a traffic light
    centroidOrigin : int
        Identifier of centroid origin of the vehicle, when the traffic
        conditions are defined by an OD matrix
    centroidDest : int
        Identifier of centroid destination of the vehicle, when the traffic
        conditions are defined by an OD matrix
    idsectionExit : int
        Identifier of exit section destination of the vehicle, when the
        destination centroid uses percentages as destination (otherwise is
        –1) and the traffic conditions are defined by an OD matrix
    idLine : int
        Identifier of Public Transport Line, when the vehicle has been
        generated as a public transport vehicle
    """

    def __init__(self):
        """Instantiate StaticInfVeh."""
        self.report = None
        self.idVeh = None
        self.type = None
        self.length = None
        self.width = None
        self.maxDesiredSpeed = None
        self.maxAcceleration = None
        self.normalDeceleration = None
        self.maxDeceleration = None
        self.speedAcceptance = None
        self.minDistanceVeh = None
        self.giveWayTime = None
        self.guidanceAcceptance = None
        self.enrouted = None
        self.equipped = None
        self.tracked = None
        self.keepfastLane = None
        self.headwayMin = None
        self.sensitivityFactor = None
        self.reactionTime = None
        self.reactionTimeAtStop = None
        self.reactionTimeAtTrafficLight = None
        self.centroidOrigin = None
        self.centroidDest = None
        self.idsectionExit = None
        self.idLine = None
