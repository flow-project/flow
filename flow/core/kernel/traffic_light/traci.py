"""Script containing the TraCI traffic light kernel class."""

from flow.core.kernel.traffic_light import KernelTrafficLight
import traci.constants as tc
from traci.exceptions import FatalTraCIError, TraCIException


class TraCITrafficLight(KernelTrafficLight):
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
        KernelTrafficLight.__init__(self, master_kernel)

        self.__tls = dict()  # contains current time step traffic light data
        self.__tls_properties = dict()  # traffic light xml properties

        # names of nodes with traffic lights
        self.__ids = []

        # number of traffic light nodes
        self.num_traffic_lights = 0

    def pass_api(self, kernel_api):
        """See parent class.

        Subscriptions and vehicle IDs are also added here.
        """
        KernelTrafficLight.pass_api(self, kernel_api)

        # names of nodes with traffic lights
        self.__ids = kernel_api.trafficlight.getIDList()

        # number of traffic light nodes
        self.num_traffic_lights = len(self.__ids)

        if self.master_kernel.vehicle.render:
            # subscribe the traffic light signal data
            for node_id in self.__ids:
                self.kernel_api.trafficlight.subscribe(
                    node_id, [tc.TL_RED_YELLOW_GREEN_STATE])

    def update(self, reset):
        """See parent class."""
        tls_obs = {}
        if self.master_kernel.vehicle.render:
            for tl_id in self.__ids:
                tls_obs[tl_id] = \
                    self.kernel_api.trafficlight.getSubscriptionResults(tl_id)
        else:
            for tl_id in self.__ids:
                tls_obs[tl_id] = self._get_libsumo_subscription_results(tl_id)
        self.__tls = tls_obs.copy()

    def get_ids(self):
        """See parent class."""
        return self.__ids

    def set_state(self, node_id, state, link_index="all"):
        """See parent class."""
        if link_index == "all":
            # if lights on all lanes are changed
            self.kernel_api.trafficlight.setRedYellowGreenState(
                tlsID=node_id, state=state)
        else:
            # if lights on a single lane is changed
            self.kernel_api.trafficlight.setLinkState(
                tlsID=node_id, tlsLinkIndex=link_index, state=state)

    def get_state(self, node_id):
        """See parent class."""
        return self.__tls[node_id][tc.TL_RED_YELLOW_GREEN_STATE]

    def _get_libsumo_subscription_results(self, tl_id):
        """Create a traci-style subscription result in the case of libsumo."""
        try:
            res = {
                tc.TL_RED_YELLOW_GREEN_STATE:
                    self.kernel_api.trafficlight.getRedYellowGreenState(tl_id)
            }
        except (TraCIException, FatalTraCIError):
            # This is in case a vehicle exited the network and has not been
            # unscubscribed yet.
            res = None

        return res
