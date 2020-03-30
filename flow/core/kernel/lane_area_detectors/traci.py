"""Script containing the TraCI lane area detector kernel class."""

from flow.core.kernel.lane_area_detectors import KernelLaneAreaDetector
import traci.constants as tc


class TraCILaneAreaDetector(KernelLaneAreaDetector):
    """Sumo traffic light kernel.

    Implements all methods discussed in the base area detector kernel class.
    """

    def __init__(self, master_kernel, variables_to_retrieve=["nVehicles"]):
        """Instantiate the sumo lane area detector kernel.

        Parameters
        ----------
        master_kernel : flow.core.kernel.Kernel
            the higher level kernel (used to call methods from other
            sub-kernels)
        """
        KernelLaneAreaDetector.__init__(self, master_kernel)

        # List of Strings variables that shall be updated while simulating
        """possible values in the list:

        "mJamLength": Jam length in meters
        "vJamLength": Jam length in vehicles
        "nHalting": Number of vehicles that were halting
        "meanSpeed": Mean speed of vehicles
        "occupancy": How much space occupied by vehicles
        "vIDs": List of vehicle ids
        "nVehicles": Number of vehicles
        """
        self.variables_to_retrieve = variables_to_retrieve

        """Static information
        """
        # names of detectors
        self.__ids = []
        # lengths of detectors
        self.__lengths = []
        # starting position of detectors in meters from beginning of the lane.
        self.__positions = []
        # ids of the lanes the detectors are on
        self.__lane_ids = []

        """Subscription information
        """
        self.__subscription_results = None

        # number of detectors
        self.num_detectors = 0

    def pass_api(self, kernel_api):
        """See parent class.

        Subscriptions and vehicle IDs are also added here.
        """
        KernelLaneAreaDetector.pass_api(self, kernel_api)

        # retrieve ids of the lane area detectors
        self.__ids = kernel_api.lanearea.getIDList()

        # retrieve static information about lane area detectors
        for det_id in self.__ids:
            self.__lengths.append(kernel_api.lanearea.getLength(det_id))
            self.__positions.append(kernel_api.lanearea.getPosition(det_id))
            self.__lane_ids.append(kernel_api.lanearea.getLaneID(det_id))

        # number of lane area detectors
        self.num_detectors = len(self.__ids)

        # initialize subscriptions
        if "mJamLength" in self.variables_to_retrieve:
            for det_id in self.__ids:
                self.kernel_api.lanearea.subscribe(
                    det_id, [tc.JAM_LENGTH_METERS])
        if "vJamLength" in self.variables_to_retrieve:
            for det_id in self.__ids:
                self.kernel_api.lanearea.subscribe(
                    det_id, [tc.JAM_LENGTH_VEHICLE])
        if "nHalting" in self.variables_to_retrieve:
            for det_id in self.__ids:
                self.kernel_api.lanearea.subscribe(
                    det_id, [tc.LAST_STEP_VEHICLE_HALTING_NUMBER])
        if "meanSpeed" in self.variables_to_retrieve:
            for det_id in self.__ids:
                self.kernel_api.lanearea.subscribe(
                    det_id, [tc.LAST_STEP_MEAN_SPEED])
        if "occupancy" in self.variables_to_retrieve:
            for det_id in self.__ids:
                self.kernel_api.lanearea.subscribe(
                    det_id, [tc.LAST_STEP_OCCUPANCY])
        if "vIDs" in self.variables_to_retrieve:
            for det_id in self.__ids:
                self.kernel_api.lanearea.subscribe(
                    det_id, [tc.LAST_STEP_VEHICLE_ID_LIST])
        if "nVehicles" in self.variables_to_retrieve:
            for det_id in self.__ids:
                self.kernel_api.lanearea.subscribe(
                    det_id, [tc.LAST_STEP_VEHICLE_NUMBER])

    def update(self, reset):
        """See parent class."""
        det_obs = {}
        for detector_id in self.__ids:
            det_obs[detector_id] = self.kernel_api.lanearea.getSubscriptionResults(detector_id)
        self.__subscription_results = det_obs.copy()

    def get_ids(self):
        """See parent class."""
        return self.__ids

    def get_n_veh_seen(self, detector_id):
        """See parent class."""
        try:
            return self.__subscription_results[detector_id][tc.LAST_STEP_VEHICLE_NUMBER]
        except ValueError:
            print("did you specify this subscription in the initialization?")

    def get_m_jam_length(self, detector_id):
        """See parent class."""
        try:
            return self.__subscription_results[detector_id][tc.JAM_LENGTH_METERS]
        except ValueError:
            print("did you specify this subscription in the initialization?")

    def get_v_jam_length(self, detector_id):
        """See parent class."""
        try:
            return self.__subscription_results[detector_id][tc.JAM_LENGTH_VEHICLE]
        except ValueError:
            print("did you specify this subscription in the initialization?")

    def get_n_halting(self, detector_id):
        """See parent class."""
        try:
            return self.__subscription_results[detector_id][tc.LAST_STEP_VEHICLE_HALTING_NUMBER]
        except ValueError:
            print("did you specify this subscription in the initialization?")

    def get_mean_speed(self, detector_id):
        """See parent class."""
        try:
            return self.__subscription_results[detector_id][tc.LAST_STEP_MEAN_SPEED]
        except ValueError:
            print("did you specify this subscription in the initialization?")

    def get_occupancy(self, detector_id):
        """See parent class."""
        try:
            return self.__subscription_results[detector_id][tc.LAST_STEP_OCCUPANCY]
        except ValueError:
            print("did you specify this subscription in the initialization?")

    def get_v_id_list(self, detector_id):
        """See parent class."""
        try:
            return self.__subscription_results[detector_id][tc.LAST_STEP_VEHICLE_ID_LIST]
        except ValueError:
            print("did you specify this subscription in the initialization?")
