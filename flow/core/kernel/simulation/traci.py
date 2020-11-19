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
            for veh_id in self.master_kernel.vehicle.get_ids():
                t = round(self.time, 2)

                # some miscellaneous pre-processing
                position = kv.get_2d_position(veh_id)

                # Make sure dictionaries corresponding to the vehicle and
                # time are available.
                if veh_id not in self.stored_data.keys():
                    self.stored_data[veh_id] = dict()
                if t not in self.stored_data[veh_id].keys():
                    self.stored_data[veh_id][t] = dict()

                # Add the speed, position, and lane data.
                self.stored_data[veh_id][t].update({
                    "speed": kv.get_speed(veh_id),
                    "lane_number": kv.get_lane(veh_id),
                    "edge_id": kv.get_edge(veh_id),
                    "relative_position": kv.get_position(veh_id),
                    "x": position[0],
                    "y": position[1],
                    "headway": kv.get_headway(veh_id),
                    "leader_id": kv.get_leader(veh_id),
                    "follower_id": kv.get_follower(veh_id),
                    "leader_rel_speed":
                        kv.get_speed(kv.get_leader(veh_id))
                        - kv.get_speed(veh_id),
                    "target_accel_with_noise_with_failsafe":
                        kv.get_accel(veh_id, noise=True, failsafe=True),
                    "target_accel_no_noise_no_failsafe":
                        kv.get_accel(veh_id, noise=False, failsafe=False),
                    "target_accel_with_noise_no_failsafe":
                        kv.get_accel(veh_id, noise=True, failsafe=False),
                    "target_accel_no_noise_with_failsafe":
                        kv.get_accel(veh_id, noise=False, failsafe=True),
                    "realized_accel":
                        kv.get_realized_accel(veh_id),
                    "road_grade": kv.get_road_grade(veh_id),
                    "distance": kv.get_distance(veh_id),
                })

    def close(self):
        """See parent class."""
        # Save the emission data to a csv.
        if self.emission_path is not None:
            self.save_emission()

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

                # ignore step logs (if requested)
                if sim_params.no_step_log:
                    sumo_call.append("--no-step-log")

                # add the lateral resolution of the sublanes (if requested)
                if sim_params.lateral_resolution is not None:
                    sumo_call.append("--lateral-resolution")
                    sumo_call.append(str(sim_params.lateral_resolution))

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
                logging.debug(" Emission file: " + str(self.emission_path))
                logging.debug(" Step length: " + str(sim_params.sim_step))

                # Opening the I/O thread to SUMO
                self.sumo_proc = subprocess.Popen(
                    sumo_call,
                    stdout=subprocess.DEVNULL
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

    def save_emission(self, run_id=0):
        """Save any collected emission data to a csv file.

        If not data was collected, nothing happens. Moreover, any internally
        stored data by this class is clear whenever data is stored.

        Parameters
        ----------
        run_id : int
            the rollout number, appended to the name of the emission file. Used
            to store emission files from multiple rollouts run sequentially.
        """
        # If there is no stored data, ignore this operation. This is to ensure
        # that data isn't deleted if the operation is called twice.
        if len(self.stored_data) == 0:
            return

        # Get a csv name for the emission file.
        name = "{}-{}_emission.csv".format(
            self.master_kernel.network.network.name, run_id)

        # The name of all stored data-points (excluding id and time)
        stored_ids = [
            "x",
            "y",
            "speed",
            "headway",
            "leader_id",
            "target_accel_with_noise_with_failsafe",
            "target_accel_no_noise_no_failsafe",
            "target_accel_with_noise_no_failsafe",
            "target_accel_no_noise_with_failsafe",
            "realized_accel",
            "road_grade",
            "edge_id",
            "lane_number",
            "distance",
            "relative_position",
            "follower_id",
            "leader_rel_speed",
        ]

        # Update the stored data to push to the csv file.
        final_data = {"time": [], "id": []}
        final_data.update({key: [] for key in stored_ids})

        for veh_id in self.stored_data.keys():
            for t in self.stored_data[veh_id].keys():
                final_data['time'].append(t)
                final_data['id'].append(veh_id)
                for key in stored_ids:
                    final_data[key].append(self.stored_data[veh_id][t][key])

        with open(os.path.join(self.emission_path, name), "w") as f:
            print(os.path.join(self.emission_path, name), self.emission_path)
            writer = csv.writer(f, delimiter=',')
            writer.writerow(final_data.keys())
            writer.writerows(zip(*final_data.values()))

        # Clear all memory from the stored data. This is useful if this
        # function is called in between resets.
        self.stored_data.clear()
