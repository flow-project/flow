"""Script containing the TraCI traffic light kernel class."""

from flow.core.kernel.induction_loops import KernelLaneAreaDetector
import traci.constants as tc


class TraCILaneAreaDetector(KernelLaneAreaDetector):
    """Sumo traffic light kernel.

    Implements all methods discussed in the base traffic light kernel class.
    """

    def __init__(self, master_kernel):
        """Instantiate the sumo traffic light kernel.

        Parameters
        ----------
        master_kernel : flow.core.kernel.Kernel
            the higher level kernel (used to call methods from other
            sub-kernels)
        """
        KernelLaneAreaDetector.__init__(self, master_kernel)

        self.__n_veh_seen = dict()  # vehicles seen per detector

        # names of detectors
        self.__ids = []

        # number of detectors
        self.num_detectors = 0

    def pass_api(self, kernel_api):
        """See parent class.

        Subscriptions and vehicle IDs are also added here.
        """
        KernelLaneAreaDetector.pass_api(self, kernel_api)

        # names of nodes with traffic lights
        self.__ids = kernel_api.lanearea.getIDList()

        # number of traffic light nodes
        self.num_detectors = len(self.__ids)

        # subscribe the traffic light signal data
        for detector_id in self.__ids:
            self.kernel_api.lanearea.subscribe(
                detector_id, [tc.LAST_STEP_VEHICLE_NUMBER])

    def initialize(self, detectors):
        for key in detectors:
            self.__ids.append(key)
        self.num_detectors = len(self.__ids)

    def update(self, reset):
        """See parent class."""
        # det_obs = {}
        # for detector_id in self.__ids:
        #     det_obs[detector_id] = \
        #         self.kernel_api.lanearea.getSubscriptionResults(detector_id)
        # self.__n_veh_seen = det_obs.copy()
        det_obs = {}
        for detector_id in self.__ids:
            det_obs[detector_id] = self.kernel_api.lanearea.getLastStepVehicleNumber(detector_id)
        self.__n_veh_seen = det_obs.copy()

    def get_ids(self):
        """See parent class."""
        return self.__ids

    def get_n_veh_seen(self, detector_id):
        """See parent class."""
        return self.__n_veh_seen[detector_id]
