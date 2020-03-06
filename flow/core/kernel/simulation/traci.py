"""Script containing the TraCI simulation kernel class."""

from flow.core.kernel.simulation import KernelSimulation
import flow.config as config
import traci.constants as tc
import traci
import traceback
import os
import time
import logging
import subprocess
import signal


# Number of retries on restarting SUMO before giving up
RETRIES_ON_ERROR = 10


class TraCISimulation(KernelSimulation):
    """Sumo simulation kernel.

    Extends flow.core.kernel.simulation.KernelSimulation
    """

    def __init__(self, master_kernel):
        """Instantiate the sumo simulator kernel.

        Parameters
        ----------
        master_kernel : flow.core.kernel.Kernel
            the higher level kernel (used to call methods from other
            sub-kernels)
        """
        KernelSimulation.__init__(self, master_kernel)
        # contains the subprocess.Popen instance used to start traci
        self.sumo_proc = None

    def pass_api(self, kernel_api):
        """See parent class.

        Also initializes subscriptions.
        """
        KernelSimulation.pass_api(self, kernel_api)

        # subscribe some simulation parameters needed to check for entering,
        # exiting, and colliding vehicles
        self.kernel_api.simulation.subscribe([
            tc.VAR_DEPARTED_VEHICLES_IDS, tc.VAR_ARRIVED_VEHICLES_IDS,
            tc.VAR_TELEPORT_STARTING_VEHICLES_IDS, tc.VAR_TIME_STEP,
            tc.VAR_DELTA_T
        ])

    def simulation_step(self):
        """See parent class."""
        self.kernel_api.simulationStep()

    def update(self, reset):
        """See parent class."""
        pass

    def close(self):
        """See parent class."""
        self.kernel_api.close()

    def check_collision(self):
        """See parent class."""
        return self.kernel_api.simulation.getStartingTeleportNumber() != 0

    def start_simulation(self, network, sim_params):
        """Start a sumo simulation instance.

        This method uses the configuration files created by the network class
        to initialize a sumo instance. Also initializes a traci connection to
        interface with sumo from Python.
        """
        error = None
        for _ in range(RETRIES_ON_ERROR):
            try:
                # port number the sumo instance will be run on
                port = sim_params.port

                sumo_binary = "sumo-gui" if sim_params.render is True \
                    else "sumo"

                # command used to start sumo
                sumo_call = [
                    sumo_binary, "-c", network.cfg,
                    "--remote-port", str(sim_params.port),
                    "--num-clients", str(sim_params.num_clients),
                ]

                logging.info(" Starting SUMO on port " + str(port))
                logging.debug(" Cfg file: " + str(network.cfg))
                if sim_params.num_clients > 1:
                    logging.info(" Num clients are" +
                                 str(sim_params.num_clients))
                logging.debug(" Step length: " + str(sim_params.sim_step))

                if sim_params.render:
                    # Opening the I/O thread to SUMO
                    self.sumo_proc = subprocess.Popen(
                        sumo_call, preexec_fn=os.setsid)

                    # wait a small period of time for the subprocess to
                    # activate before trying to connect with traci
                    if os.environ.get("TEST_FLAG", 0):
                        time.sleep(0.1)
                    else:
                        time.sleep(config.SUMO_SLEEP)

                    traci_connection = traci.connect(port, numRetries=100)
                    traci_connection.setOrder(0)
                    traci_connection.simulationStep()
                else:
                    import libsumo

                    # Use libsumo to create a simulation instance.
                    libsumo.start(sumo_call[1:3])
                    libsumo.simulationStep()

                    # libsumo will act as the kernel API
                    traci_connection = libsumo

                return traci_connection
            except Exception as e:
                print("Error during start: {}".format(traceback.format_exc()))
                error = e
                self.teardown_sumo()
        raise error

    def teardown_sumo(self):
        """Kill the sumo subprocess instance."""
        try:
            # In case not using libsumo, kill the process.
            if self.sumo_proc is not None:
                os.killpg(self.sumo_proc.pid, signal.SIGTERM)
        except Exception as e:
            print("Error during teardown: {}".format(e))
