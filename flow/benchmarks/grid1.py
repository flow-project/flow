"""Benchmark for grid1.

Action Dimension: (25, )

Observation Dimension: (915, )

Horizon: 400 steps
"""

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams
from flow.core.vehicles import Vehicles
from flow.controllers import SumoCarFollowingController, GridRouter

# time horizon of a single rollout
HORIZON = 400
# inflow rate of vehicles at every edge
EDGE_INFLOW = 300
# enter speed for departing vehicles
V_ENTER = 30
# number of row of bidirectional lanes
N_ROWS = 5
# number of columns of bidirectional lanes
N_COLUMNS = 5
# length of inner edges in the grid network
INNER_LENGTH = 300
# length of final edge in route
LONG_LENGTH = 100
# length of edges that vehicles start on
SHORT_LENGTH = 300
# number of vehicles originating in the left, right, top, and bottom edges
N_LEFT, N_RIGHT, N_TOP, N_BOTTOM = 1, 1, 1, 1

# we place a sufficient number of vehicles to ensure they confirm with the
# total number specified above. We also use a "right_of_way" speed mode to
# support traffic light compliance
vehicles = Vehicles()
vehicles.add(
    veh_id="human",
    acceleration_controller=(SumoCarFollowingController, {}),
    sumo_car_following_params=SumoCarFollowingParams(
        min_gap=2.5,
        max_speed=V_ENTER,
        speed_mode="right_of_way",
    ),
    routing_controller=(GridRouter, {}),
    num_vehicles=(N_LEFT + N_RIGHT) * N_COLUMNS + (N_BOTTOM + N_TOP) * N_ROWS)

# inflows of vehicles are place on all outer edges (listed here)
outer_edges = []
outer_edges += ["left{}_{}".format(N_ROWS, i) for i in range(N_COLUMNS)]
outer_edges += ["right0_{}".format(i) for i in range(N_ROWS)]
outer_edges += ["bot{}_0".format(i) for i in range(N_ROWS)]
outer_edges += ["top{}_{}".format(i, N_COLUMNS) for i in range(N_ROWS)]

# equal inflows for each edge (as dictate by the EDGE_INFLOW constant)
inflow = InFlows()
for edge in outer_edges:
    inflow.add(
        veh_type="human",
        edge=edge,
        vehs_per_hour=EDGE_INFLOW,
        departLane="free",
        departSpeed="max")

flow_params = dict(
    # name of the experiment
    exp_tag="grid_1",

    # name of the flow environment the experiment is running on
    env_name="PO_TrafficLightGridEnv",

    # name of the scenario class the experiment is running on
    scenario="SimpleGridScenario",

    # name of the generator used to create/modify network configuration files
    generator="SimpleGridGenerator",

    # sumo-related parameters (see flow.core.params.SumoParams)
    sumo=SumoParams(
        restart_instance=True,
        sim_step=1,
        render=False,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        additional_params={
            "target_velocity": 50,
            "switch_time": 2,
            "num_observed": 2,
            "discrete": False,
            "tl_type": "actuated"
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
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
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.vehicles.Vehicles)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(shuffle=True),
)
