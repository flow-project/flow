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
        self.num_meters = 0

    def pass_api(self, kernel_api):
        """See parent class."""
        self.kernel_api = kernel_api

    def update(self, reset):
        """See parent class."""
        pass

    def get_ids(self):
        """See parent class."""
        return self.kernel_api.get_traffic_light_ids()

    def set_state(self, meter_aimsun_id, state):
        """Set the state of the traffic lights on a specific meter.

        Parameters
        ----------
        meter_aimsun_id : int
            aimsun id of the meter
        state : int
            desired state(s) for the traffic light
            0: red
            1: green
            2: yellow
        link_index : int, optional
            index of the link whose traffic light state is meant to be changed.
            If no value is provided, the lights on all links are updated.
        """
        self.kernel_api.set_traffic_light_state(meter_aimsun_id, None, state)

    def get_state(self, meter_aimsun_id):
        """Return the state of the traffic light(s) at the specified node.

        Parameters
        ----------
        meter_aimsun_id: int
            aimsun id of the meter

        Returns
        -------
        state : int
            desired state(s) for the traffic light
            0: red
            1: green
            2: yellow
        """
        return self.kernel_api.get_traffic_light_state(meter_aimsun_id)
