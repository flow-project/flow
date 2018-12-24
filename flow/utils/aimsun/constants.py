"""Constants used by the aimsun API for sending/receiving TCP messages."""

###############################################################################
#                             Simulation Commands                             #
###############################################################################

# simulation step
SIMULATION_STEP = 0x00

# terminate the simulation
SIMULATION_TERMINATE = 0x01

# check for collision in simulation TODO: check
SIMULATION_COLLISION = None


###############################################################################
#                              Scenario Commands                              #
###############################################################################


###############################################################################
#                               Vehicle Commands                              #
###############################################################################

# add a vehicle
ADD_VEHICLE = 0x02

# remove a vehicle
REMOVE_VEHICLE = 0x03

# set vehicle speed
VEH_SET_SPEED = 0x04

# apply vehicle lane change
VEH_SET_LANE = 0x05

# set vehicle route
VEH_SET_ROUTE = 0x06

# set color
VEH_SET_COLOR = 0x07

# get vehicle IDs
VEH_GET_IDS = 0x08

# get vehicle static information
VEH_GET_STATIC = 0x09

# get vehicle tracking information
VEH_GET_TRACKING = 0x0A

# get vehicle leader
VEH_GET_LEADER = None

# get vehicle follower
VEH_GET_FOLLOWER = None

# get vehicle route
VEH_GET_ROUTE = None

# get vehicle speed if no API command was submitted TODO: check
VEH_GET_DEFAULT_SPEED = None

# TODO: not 100% sure what this is...
VEH_GET_ORIENTATION = None

# TODO: not 100% sure what this is...
VEH_GET_TIMESTEP = None

# TODO: not 100% sure what this is...
VEH_GET_TIMEDELTA = None


###############################################################################
#                           Traffic Light Commands                            #
###############################################################################

# get traffic light IDs
TL_GET_IDS = None

# set traffic light state
TL_SET_STATE = None

# get traffic light state
TL_GET_STATE = None
