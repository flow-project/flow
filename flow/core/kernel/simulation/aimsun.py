"""Script containing the base simulation kernel class."""
from flow.core.kernel.simulation.base import KernelSimulation
from flow.utils.aimsun.api import FlowAimsunAPI
import subprocess
import os.path as osp
import csv
from flow.core.util import ensure_dir

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
        self.emission_path = None

        # used to internally keep track of the simulation time
        self.time = 0

        # a file used to store data if an emission file is provided
        self.stored_data = {
            'time': [],
            # 'x': [],
            # 'y': [],
            # 'angle': [],
            'type': [],
            'id': [],
            'relative_position': [],
            'speed': [],
            'edge_id': [],
            'lane_number': []
        }

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

        self.emission_path = sim_params.emission_path
        if self.emission_path is not None:
            ensure_dir(self.emission_path)

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

        return FlowAimsunAPI(port=sim_params.port)

    def simulation_step(self):
        """See parent class."""
        self.kernel_api.simulation_step()

    def update(self, reset):
        """See parent class.

        No update is needed in this case.
        """
        if reset:
            self.time = 0
        else:
            self.time += self.sim_step

        if self.emission_path is not None:
            for veh_id in self.master_kernel.vehicle.get_ids():
                self.stored_data['id'].append(
                    veh_id)
                self.stored_data['time'].append(
                    self.time)
                self.stored_data['type'].append(
                    self.master_kernel.vehicle.get_type(veh_id))
                # self.stored_data['x'].append()
                # self.stored_data['y'].append()
                self.stored_data['relative_position'].append(
                    self.master_kernel.vehicle.get_position(veh_id))
                # self.stored_data['angle'].append()
                self.stored_data['speed'].append(
                    self.master_kernel.vehicle.get_speed(veh_id))
                self.stored_data['edge_id'].append(
                    self.master_kernel.vehicle.get_edge(veh_id))
                self.stored_data['lane_number'].append(
                    self.master_kernel.vehicle.get_lane(veh_id))

    def check_collision(self):
        """See parent class."""
        return False

    def close(self):
        """See parent class."""
        # save the emission data to a csv
        if self.emission_path is not None:
            name = "%s_emission.csv" % self.master_kernel.scenario.network.name
            with open(osp.join(self.emission_path, name), "w") as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(self.stored_data.keys())
                writer.writerows(zip(*self.stored_data.values()))

        try:
            self.kernel_api.stop_simulation()
            self.aimsun_proc.kill()
        except OSError:
            # in case no simulation originally existed (used by the visualizer)
            pass
