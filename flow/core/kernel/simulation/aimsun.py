"""Script containing the base simulation kernel class."""
from flow.core.kernel.simulation.base import KernelSimulation
from flow.utils.aimsun.api import FlowAimsunAPI
import subprocess
import os.path as osp

try:
    # Load user config if exists, else load default config
    import flow.config as config
except ImportError:
    import flow.config_default as config


class AimsunKernelSimulation(KernelSimulation):
    """Base simulation kernel.

    The simulation kernel is responsible for generating the simulation and
    passing to all other kernel the API that they can use to interact with the
    simulation.

    The simulation kernel is also responsible for advancing, resetting, and
    storing whatever simulation data is relevant.

    All methods in this class are abstract and must be overwritten by other
    child classes.
    """

    def __init__(self, master_kernel):
        """Initialize the Aimsun simulation kernel."""
        KernelSimulation.__init__(self, master_kernel)

        self.master_kernel = master_kernel
        self.kernel_api = None
        self.sim_step = None
        self.aimsun_proc = None

    def pass_api(self, kernel_api):
        """See parent class."""
        self.kernel_api = kernel_api

    def start_simulation(self, network, sim_params):
        """See parent class.

        This method calls the aimsun generator to generate the network, starts
        a simulation, and creates a class to communicate with the simulation
        via an TCP connection.
        """
        # FIXME: hack
        sim_params.port = 9999

        # save the simulation step size (for later use)
        self.sim_step = sim_params.sim_step

        # path to the Aimsun_Next binary
        aimsun_path = osp.join(osp.expanduser(config.AIMSUN_NEXT_PATH),
                               'Aimsun_Next')

        # path to the supplementary file that is used to generate an aimsun
        # network from a template
        script_path = osp.join(config.PROJECT_PATH,
                               'flow/utils/aimsun/generate.py')

        # start the aimsun process
        aimsun_call = [aimsun_path, "-script", script_path]
        self.aimsun_proc = subprocess.Popen(aimsun_call)

        return None
        # return FlowAimsunAPI(port=sim_params.port)

    def simulation_step(self):
        """See parent class."""
        self.kernel_api.simulation_step()

    def update(self, reset):
        """See parent class.

        No update is needed in this case.
        """
        pass

    def check_collision(self):
        """See parent class."""
        veh_ids = self.master_kernel.vehicle.get_ids()
        for veh in veh_ids:
            headway = self.master_kernel.vehicle.get_headway(veh)
            if headway <= 0:
                return True
        return False

    def close(self):
        """See parent class."""
        self.kernel_api.stop_simulation()
        self.aimsun_proc.kill()
