"""Script containing the TraCI simulation kernel class."""

from flow.core.kernel.simulation import KernelSimulation
from flow.core.util import ensure_dir
import flow.config as config
import traci.constants as tc
import traci
import traceback
import os
import time
import logging
import subprocess
import signal
import csv


# Number of retries on restarting SUMO before giving up
RETRIES_ON_ERROR = 10


class TraCISimulation(KernelSimulation):
    """Sumo simulation kernel.

    Extends flow.core.kernel.simulation.KernelSimulation

    Attributes
    ----------
    sumo_proc : subprocess.Popen
        contains the subprocess.Popen instance used to start traci
    sim_step : float
        seconds per simulation step
    emission_path : str or None
        Path to the folder in which to create the emissions output. Emissions
        output is not generated if this value is not specified
    time : float
        used to internally keep track of the simulation time
    stored_data : dict <str, dict <float, dict <str, Any>>>
        a dict object used to store additional data if an emission file is
        provided. The first key corresponds to the name of the vehicle, the
        second corresponds to the time the sample was issued, and the final
        keys represent the additional data stored at every given time for every
        vehicle, and consists of the following keys:

        * acceleration (no noise): the accelerations issued to the vehicle,
          excluding noise
        * acceleration (requested): the requested acceleration by the vehicle,
          including noise
        * acceleration (actual): the actual acceleration by the vehicle,
          collected by computing the difference between the speeds of the
          vehicle and dividing it by the sim_step term
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

        self.sumo_proc = None
        self.sim_step = None
        self.emission_path = None
        self.time = 0
        self.stored_data = dict()

    def pass_api(self, kernel_api):
        """See parent class.

        Also initializes subscriptions.
        """
        KernelSimulation.pass_api(self, kernel_api)

        # subscribe some simulation parameters needed to check for entering,
        # exiting, and colliding vehicles
        self.kernel_api.simulation.subscribe([
            tc.VAR_DEPARTED_VEHICLES_IDS,
            tc.VAR_ARRIVED_VEHICLES_IDS,
            tc.VAR_TELEPORT_STARTING_VEHICLES_IDS,
            tc.VAR_TIME_STEP,
            tc.VAR_DELTA_T,
            tc.VAR_LOADED_VEHICLES_NUMBER,
            tc.VAR_DEPARTED_VEHICLES_NUMBER,
            tc.VAR_ARRIVED_VEHICLES_NUMBER
        ])

    def simulation_step(self):
        """See parent class."""
        self.kernel_api.simulationStep()

    def update(self, reset):
        """See parent class."""
        if reset:
            self.time = 0
        else:
            self.time += self.sim_step

        # Collect the additional data to store in the emission file.
        if self.emission_path is not None:
            kv = self.master_kernel.vehicle
            dt = self.sim_step
            for veh_id in self.master_kernel.vehicle.get_ids():
                t = round(self.time, 2)

                # Make sure dictionaries corresponding to the vehicle and
                # time are available.
                if veh_id not in self.stored_data.keys():
                    self.stored_data[veh_id] = dict()
                if t not in self.stored_data[veh_id].keys():
                    self.stored_data[veh_id][t] = dict()

                # Add the speed, position, and lane data.
                self.stored_data[veh_id][t]['speed'] = \
                    kv.get_speed(veh_id)
                self.stored_data[veh_id][t]['lane_number'] = \
                    kv.get_lane(veh_id)
                self.stored_data[veh_id][t]['edge_id'] = \
                    kv.get_edge(veh_id)
                self.stored_data[veh_id][t]['relative_position'] = \
                    kv.get_position(veh_id)

                # Add the acceleration data.
                t = round(self.time - dt, 2)
                if not reset and t in self.stored_data[veh_id].keys():
                    self.stored_data[veh_id][t]['acceleration (no noise)'] = \
                        kv.get_previous_acceleration(veh_id, noise=False)
                    self.stored_data[veh_id][t]['acceleration (requested)'] = \
                        kv.get_previous_acceleration(veh_id)
                    self.stored_data[veh_id][t]['acceleration (actual)'] = \
                        (kv.get_speed(veh_id) -
                         kv.get_previous_speed(veh_id)) / dt

    def close(self):
        """See parent class."""
        # Save the emission data to a csv.
        if self.emission_path is not None:
            # Get a csv name for the emission file.
            name = "%s_emission.csv" % self.master_kernel.network.network.name

            # Update the stored data to push to the csv file.
            final_data = {
                'id': [],
                'time': [],
                'speed': [],
                'lane_number': [],
                'edge_id': [],
                'relative_position': [],
                'acceleration (no noise)': [],
                'acceleration (requested)': [],
                'acceleration (actual)': [],
            }

            for veh_id in self.stored_data.keys():
                for t in self.stored_data[veh_id].keys():
                    try:
                        final_data['id'].append(
                            veh_id)
                        final_data['time'].append(
                            t)
                        final_data['speed'].append(
                            self.stored_data[veh_id][t]['speed'])
                        final_data['lane_number'].append(
                            self.stored_data[veh_id][t]['lane_number'])
                        final_data['edge_id'].append(
                            self.stored_data[veh_id][t]['edge_id'])
                        final_data['relative_position'].append(
                            self.stored_data[veh_id][t]['relative_position'])
                        final_data['acceleration (no noise)'].append(
                            self.stored_data[veh_id][t].get(
                                'acceleration (no noise)', 0))
                        final_data['acceleration (requested)'].append(
                            self.stored_data[veh_id][t].get(
                                'acceleration (requested)', 0))
                        final_data['acceleration (actual)'].append(
                            self.stored_data[veh_id][t].get(
                                'acceleration (actual)', 0))
                    except KeyError:
                        # To handle vehicles that just entered
                        pass

            with open(os.path.join(self.emission_path, name), "w") as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(final_data.keys())
                writer.writerows(zip(*final_data.values()))

        self.kernel_api.close()

    def check_collision(self):
        """See parent class."""
        return self.kernel_api.simulation.getStartingTeleportNumber() != 0

    def start_simulation(self, network, sim_params):
        """Start a sumo simulation instance.

        This method performs the following operations:

        1. It collect the simulation step size and the emission path
           information. If an emission path is specifies, it ensures that the
           path exists.
        2. It also uses the configuration files created by the network class to
           initialize a sumo instance.
        3. Finally, It initializes a traci connection to interface with sumo
           from Python and returns the connection.
        """
        # Save the simulation step size (for later use).
        self.sim_step = sim_params.sim_step

        # Update the emission path term.
        self.emission_path = sim_params.emission_path
        if self.emission_path is not None:
            ensure_dir(self.emission_path)

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
                    "--step-length", str(sim_params.sim_step)
                ]

                # use a ballistic integration step (if request)
                if sim_params.use_ballistic:
                    sumo_call.append("--step-method.ballistic")

                # add step logs (if requested)
                if sim_params.no_step_log:
                    sumo_call.append("--no-step-log")

                # add the lateral resolution of the sublanes (if requested)
                if sim_params.lateral_resolution is not None:
                    sumo_call.append("--lateral-resolution")
                    sumo_call.append(str(sim_params.lateral_resolution))

                # # add the emission path to the sumo command (if requested)
                # if sim_params.emission_path is not None:
                #     ensure_dir(sim_params.emission_path)
                #     emission_out = os.path.join(
                #         sim_params.emission_path,
                #         "{0}-emission.xml".format(network.name))
                #     sumo_call.append("--emission-output")
                #     sumo_call.append(emission_out)
                # else:
                #     emission_out = None

                if sim_params.overtake_right:
                    sumo_call.append("--lanechange.overtake-right")
                    sumo_call.append("true")

                # specify a simulation seed (if requested)
                if sim_params.seed is not None:
                    sumo_call.append("--seed")
                    sumo_call.append(str(sim_params.seed))

                if not sim_params.print_warnings:
                    sumo_call.append("--no-warnings")
                    sumo_call.append("true")

                # set the time it takes for a gridlock teleport to occur
                sumo_call.append("--time-to-teleport")
                sumo_call.append(str(int(sim_params.teleport_time)))

                # check collisions at intersections
                sumo_call.append("--collision.check-junctions")
                sumo_call.append("true")

                logging.info(" Starting SUMO on port " + str(port))
                logging.debug(" Cfg file: " + str(network.cfg))
                if sim_params.num_clients > 1:
                    logging.info(" Num clients are" +
                                 str(sim_params.num_clients))
                # logging.debug(" Emission file: " + str(emission_out))
                logging.debug(" Step length: " + str(sim_params.sim_step))

                # Opening the I/O thread to SUMO
                self.sumo_proc = subprocess.Popen(
                    sumo_call,
                    stdout=subprocess.DEVNULL,
                    preexec_fn=os.setsid
                )

                # wait a small period of time for the subprocess to activate
                # before trying to connect with traci
                if os.environ.get("TEST_FLAG", 0):
                    time.sleep(0.1)
                else:
                    time.sleep(config.SUMO_SLEEP)

                traci_connection = traci.connect(port, numRetries=100)
                traci_connection.setOrder(0)
                traci_connection.simulationStep()

                return traci_connection
            except Exception as e:
                print("Error during start: {}".format(traceback.format_exc()))
                error = e
                self.teardown_sumo()
        raise error

    def teardown_sumo(self):
        """Kill the sumo subprocess instance."""
        try:
            os.killpg(self.sumo_proc.pid, signal.SIGTERM)
        except Exception as e:
            print("Error during teardown: {}".format(e))
