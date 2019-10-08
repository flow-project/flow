"""Constants used by the aimsun API for sending/receiving TCP messages."""

###############################################################################
#                             Simulation Commands                             #
###############################################################################

#: simulation step
SIMULATION_STEP = 0x00

#: terminate the simulation
SIMULATION_TERMINATE = 0x01


###############################################################################
#                               Network Commands                              #
###############################################################################

#: get the edge name in aimsun
GET_EDGE_NAME = 0x02


###############################################################################
#                               Vehicle Commands                              #
###############################################################################

#: add a vehicle
ADD_VEHICLE = 0x03

#: remove a vehicle
REMOVE_VEHICLE = 0x04

#: set vehicle speed
VEH_SET_SPEED = 0x05

#: apply vehicle lane change
VEH_SET_LANE = 0x06

#: set vehicle route
VEH_SET_ROUTE = 0x07

#: set color
VEH_SET_COLOR = 0x08

#: get IDs of entering vehicles
VEH_GET_ENTERED_IDS = 0x09

#: get IDs of exiting vehicles
VEH_GET_EXITED_IDS = 0x0A

#: get vehicle type in Aimsun
VEH_GET_TYPE_ID = 0x0B

#: get vehicle static information
VEH_GET_STATIC = 0x0C

#: get vehicle tracking information
VEH_GET_TRACKING = 0x0D

#: get vehicle leader
VEH_GET_LEADER = 0x0E

#: get vehicle follower
VEH_GET_FOLLOWER = 0x0F

#: get vehicle next section
VEH_GET_NEXT_SECTION = 0x10

#: get vehicle route
VEH_GET_ROUTE = 0x11

#: get vehicle speed if no API command was submitted
VEH_GET_DEFAULT_SPEED = 0x12

#: get vehicle angle
VEH_GET_ORIENTATION = 0x13

# TODO: not 100% sure what this is...
VEH_GET_TIMESTEP = 0x14

# TODO: not 100% sure what this is...
VEH_GET_TIMEDELTA = 0x15

#: get vehicle type name in Aimsun
VEH_GET_TYPE_NAME = 0x16

#: get vehicle length
VEH_GET_LENGTH = 0x17

#: set vehicle as tracked in Aimsun
VEH_SET_TRACKED = 0x18

#: set vehicle as untracked in Aimsun
VEH_SET_NO_TRACKED = 0x19


###############################################################################
#                           Traffic Light Commands                            #
###############################################################################

#: get traffic light IDs
TL_GET_IDS = 0x1A

#: set traffic light state
TL_SET_STATE = 0x1B

#: get traffic light state
TL_GET_STATE = 0x1C
