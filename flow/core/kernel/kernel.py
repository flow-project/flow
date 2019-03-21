"""Script containing the Flow kernel object for interacting with simulators."""

from flow.core.kernel.simulation import TraCISimulation, AimsunKernelSimulation
from flow.core.kernel.scenario import TraCIScenario, AimsunKernelScenario
from flow.core.kernel.vehicle import TraCIVehicle, AimsunKernelVehicle
from flow.core.kernel.traffic_light import TraCITrafficLight, \
    AimsunKernelTrafficLight
from flow.utils.exceptions import FatalFlowError


class Kernel(object):
    """Kernel for abstract function calling across traffic simulator APIs.

    The kernel contains four different subclasses for distinguishing between
    the various components of a traffic simulator.

    * simulation: controls starting, loading, saving, advancing, and resetting
      a simulation in Flow (see flow/core/kernel/simulation/base.py)
    * scenario: stores network-specific information (see
      flow/core/kernel/scenario/base.py)
    * vehicle: stores and regularly updates vehicle-specific information. At
      times, this class is optimized to efficiently collect information from
      the simulator (see flow/core/kernel/vehicle/base.py).
    * traffic_light: stores and regularly updates traffic light-specific
      information (see flow/core/kernel/traffic_light/base.py).

    The above kernel subclasses are designed specifically to support
    simulator-agnostic state information calling. For example, if you would
    like to collect the vehicle speed of a specific vehicle, then simply type:

    >>> k = Kernel(simulator="...")  # a kernel for some simulator type
    >>> veh_id = "..."  # some vehicle ID
    >>> speed = k.vehicle.get_speed(veh_id)

    In addition, these subclasses support sending commands to the simulator via
    its API. For example, in order to assign a specific vehicle a target
    acceleration, type:

    >>> k = Kernel(simulator="...")  # a kernel for some simulator type
    >>> veh_id = "..."  # some vehicle ID
    >>> k.vehicle.apply_acceleration(veh_id)

    These subclasses can be modified and recycled to support various different
    traffic simulators, e.g. SUMO, AIMSUN, TruckSim, etc...
    """

    def __init__(self, simulator, sim_params):
        """Instantiate a Flow kernel object.

        Parameters
        ----------
        simulator : str
            simulator type, must be one of {"traci"}
        sim_params : flow.core.params.SimParams
            simulation-specific parameters

        Raises
        ------
        flow.utils.exceptions.FatalFlowError
            if the specified input simulator is not a valid type
        """
        self.kernel_api = None

        if simulator == "traci":
            self.simulation = TraCISimulation(self)
            self.scenario = TraCIScenario(self, sim_params)
            self.vehicle = TraCIVehicle(self, sim_params)
            self.traffic_light = TraCITrafficLight(self)
        elif simulator == 'aimsun':
            self.simulation = AimsunKernelSimulation(self)
            self.scenario = AimsunKernelScenario(self, sim_params)
            self.vehicle = AimsunKernelVehicle(self, sim_params)
            self.traffic_light = AimsunKernelTrafficLight(self)
        else:
            raise FatalFlowError('Simulator type "{}" is not valid.'.
                                 format(simulator))

    def pass_api(self, kernel_api):
        """Pass the kernel API to all kernel subclasses."""
        self.kernel_api = kernel_api
        self.simulation.pass_api(kernel_api)
        self.scenario.pass_api(kernel_api)
        self.vehicle.pass_api(kernel_api)
        self.traffic_light.pass_api(kernel_api)

    def update(self, reset):
        """Update the kernel subclasses after a simulation step.

        This is meant to support optimizations in the performance of some
        simulators. For example, this step allows the vehicle subclass in the
        "traci" simulator uses the ``update`` method to collect and store
        subscription information.

        Parameters
        ----------
        reset : bool
            specifies whether the simulator was reset in the last simulation
            step
        """
        self.vehicle.update(reset)
        self.traffic_light.update(reset)
        self.scenario.update(reset)
        self.simulation.update(reset)

    def close(self):
        """Terminate all components within the simulation and scenario."""
        self.scenario.close()
        self.simulation.close()
