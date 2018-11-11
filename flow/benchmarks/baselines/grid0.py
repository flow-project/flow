"""Evaluates the baseline performance of grid0 without RL control.

Baseline is an actuated traffic light provided by SUMO.
"""

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams
from flow.core.vehicles import Vehicles
from flow.core.traffic_lights import TrafficLights
from flow.controllers import SumoCarFollowingController, GridRouter
from flow.envs.green_wave_env import PO_TrafficLightGridEnv
from flow.core.experiment import SumoExperiment
from flow.scenarios.grid import SimpleGridScenario
import numpy as np

# time horizon of a single rollout
HORIZON = 400
# inflow rate of vehicles at every edge
EDGE_INFLOW = 300
# enter speed for departing vehicles
V_ENTER = 30
# number of row of bidirectional lanes
N_ROWS = 3
# number of columns of bidirectional lanes
N_COLUMNS = 3
# length of inner edges in the grid network
INNER_LENGTH = 300
# length of final edge in route
LONG_LENGTH = 100
# length of edges that vehicles start on
SHORT_LENGTH = 300
# number of vehicles originating in the left, right, top, and bottom edges
N_LEFT, N_RIGHT, N_TOP, N_BOTTOM = 1, 1, 1, 1
# how long the yellow phase should last at a minimum
YELLOW_TIME = 3



def grid0_baseline(num_runs, render=True):
    """Run script for the grid0 baseline.

    Parameters
    ----------
        num_runs : int
            number of rollouts the performance of the environment is evaluated
            over
        render : bool, optional
            specifies whether to use sumo's gui during execution

    Returns
    -------
        SumoExperiment
            class needed to run simulations
    """
    # we place a sufficient number of vehicles to ensure they confirm with the
    # total number specified above. We also use a "right_of_way" speed mode to
    # support traffic light compliance
    vehicles = Vehicles()
    vehicles.add(veh_id="human",
                 acceleration_controller=(SumoCarFollowingController, {}),
                 sumo_car_following_params=SumoCarFollowingParams(
                     max_speed=V_ENTER,
                 ),
                 routing_controller=(GridRouter, {}),
                 num_vehicles=(N_LEFT+N_RIGHT)*N_COLUMNS +
                              (N_BOTTOM+N_TOP)*N_ROWS,
                 speed_mode="right_of_way")

    # inflows of vehicles are place on all outer edges (listed here)
    outer_edges = []
    outer_edges += ["left{}_{}".format(N_ROWS, i) for i in range(N_COLUMNS)]
    outer_edges += ["right0_{}".format(i) for i in range(N_ROWS)]
    outer_edges += ["bot{}_0".format(i) for i in range(N_ROWS)]
    outer_edges += ["top{}_{}".format(i, N_COLUMNS) for i in range(N_ROWS)]

    # equal inflows for each edge (as dictate by the EDGE_INFLOW constant)
    inflow = InFlows()
    for edge in outer_edges:
        inflow.add(veh_type="human", edge=edge, vehs_per_hour=EDGE_INFLOW,
                   departLane="free", departSpeed=V_ENTER)

    # define the traffic light logic
    tl_logic = TrafficLights(baseline=False)
    program_id = 1
    max_gap = 3.0
    detector_gap = 0.8
    show_detectors = True
    phases = [{
        "duration": "31",
        "minDur": "8",
        "maxDur": "45",
        "state": "GrGr"
    }, {
        "duration": "6",
        "minDur": str(YELLOW_TIME),
        "maxDur": "6",
        "state": "yryr"
    }, {
        "duration": "31",
        "minDur": "8",
        "maxDur": "45",
        "state": "rGrG"
    }, {
        "duration": "6",
        "minDur": str(YELLOW_TIME),
        "maxDur": "6",
        "state": "ryry"
    }]

    for i in range(N_COLUMNS * N_ROWS):
        tl_logic.add('center'+str(i), tls_type="actuated",
                     programID=program_id, phases=phases,
                     maxGap=max_gap, detectorGap=detector_gap,
                     showDetectors=show_detectors)

    net_params = NetParams(
            inflows=inflow,
            no_internal_links=False,
            additional_params={
                "speed_limit": V_ENTER + 5,
                "grid_array": {
                    "short_length": SHORT_LENGTH,
                    "inner_length": INNER_LENGTH,
                    "long_length": LONG_LENGTH,
                    "row_num": N_ROWS,
                    "col_num": N_COLUMNS,
                    "cars_left": N_LEFT,
                    "cars_right": N_RIGHT,
                    "cars_top": N_TOP,
                    "cars_bot": N_BOTTOM,
                },
                "horizontal_lanes": 1,
                "vertical_lanes": 1,
            },
        )

    sumo_params = SumoParams(
            restart_instance=False,
            sim_step=1,
            render=render,
        )

    env_params = EnvParams(
            evaluate=True,  # Set to True to evaluate traffic metrics
            horizon=HORIZON,
            additional_params={
                "target_velocity": 50,
                "switch_time": YELLOW_TIME,
                "num_observed": 2,
                "discrete": False,
                "tl_type": "actuated"
            },
        )

    initial_config = InitialConfig(shuffle=True)

    scenario = SimpleGridScenario(name="grid",
                                  vehicles=vehicles,
                                  net_params=net_params,
                                  initial_config=initial_config,
                                  traffic_lights=tl_logic)

    env = PO_TrafficLightGridEnv(env_params, sumo_params, scenario)

    exp = SumoExperiment(env, scenario)

    results = exp.run(num_runs, HORIZON)
    total_delay = np.mean(results["returns"])

    return total_delay


if __name__ == "__main__":
    runs = 1  # number of simulations to average over
    res = grid0_baseline(num_runs=runs)

    print('---------')
    print('The total delay across {} runs is {}'.format(runs, res))
