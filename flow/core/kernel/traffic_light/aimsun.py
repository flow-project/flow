"""Script containing the base traffic light kernel class."""
from flow.core.kernel.traffic_light.base import KernelTrafficLight


class AimsunKernelTrafficLight(KernelTrafficLight):
    """Aimsun traffic light kernel.

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

        # names of nodes with traffic lights
        self.__ids = []

    def pass_api(self, kernel_api):
        """See parent class."""
        self.kernel_api = kernel_api
        self.__ids = []
        # self.__ids = self.kernel_api.get_traffic_light_ids()

    def update(self, reset):
        """See parent class."""
        pass

    def get_ids(self):
        """See parent class."""
        return self.__ids

    def set_state(self, node_id, state, link_index="all"):
        """Set the state of the traffic lights on a specific node.

        Parameters
        ----------
        node_id : str
            name of the node with the controlled traffic lights
        state : str
            desired state(s) for the traffic light
            0: red
            1: green
            2: yellow
        link_index : int, optional
            index of the link whose traffic light state is meant to be changed.
            If no value is provided, the lights on all links are updated.
        """
        time = self.kernel_api.AKIGetCurrentSimulationTime()  # simulation time
        sim_step = self.master_kernel.simulation.sim_step
        identity = 0  # TODO double check
        self.kernel_api.ECIChangeStateMeteringById(
            int(node_id), state, time, sim_step, identity)

    def get_state(self, node_id):
        """Return the state of the traffic light(s) at the specified node.

        Parameters
        ----------
        node_id: str
            name of the node

        Returns
        -------
        state : str
            Index = lane index
            Element = state of the traffic light at that node/lane
        """
        # FIXME: pass all
        return None
        # return self.kernel_api.ECIGetCurrentStateofMeteringById(
        #         int(node_id), int(lane_id))

    @property
    def num_traffic_lights(self):
        """See parent class."""
        return len(self.__ids)
