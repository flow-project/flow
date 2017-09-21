cistar tutorial
******************

1. Introduction
===============

This serves as an introduction to creating your first experiment on
cistar. Cistar is a framework for deep reinforcement learning in
mixed-autonomy traffic scenarios, and it interfaces RL library ``rllab``
with traffic microsimulator ``SUMO``. Through this framework, autonomous
vehicles may be trained to perform various tasks that improve the
performance of traffic networks. Currently, cistar v0.1 supports the
implementation of simple closed networks, such as ring roads, figure
eights, etc... In order to run an experiment on cistar, three primary
classes are required:

-  **Generator**: Generates the necessary configuration files to create
   a transportation network with SUMO.
-  **Scenario**: Specifies the location of edge nodes in the network,
   and the positioning of vehicles at the start of a run.
-  **Environment**: ties the components of SUMO and rllab together,
   running a system of vehicles in a network over discrete time steps,
   while treating some of these vehicles as reinforcement learning
   agents whose actions are specified by rllab.

Once the above classes are ready, an **experiment** may be prepared to
run the environment for various levels of autonomous vehicle penetration
ranging from 0% to 100%.

In this tutorial, we will create a simple ring road network, in which
autonomous vehicles are trained to allow other vehicles in the network
to move as fast as possible. The remainder of the tutorial is organized
as follows:

-  In Sections 2, 3, and 4, we create the primary classes needed to run
   a ring road experiment.
-  In Section 5, we create a task to run the experiment in the absence
   of autonomous vehicles, and witness the performance of the network in
   this control case.
-  In Section 6, we create a task that runs the experiment with the
   inclusion of autonomous vehicles, and discuss the changes that take
   place once the reinforcement learning algorithm has converged.

2 Creating a Generator
======================

A generator class prepares the configuration files needed to create a
transportation network in sumo. A transportation network can be thought
of as a directed graph consisting of nodes, edges, routes, and other
(optional) elements.

2.1 Inheriting the Base Generator
---------------------------------

Cistar contains an abstract generator class in
``cistar_dev/core/generator.py``. In order to design any desired network
shape, a child class is created from the base generator. The base
generator class accepts a single input: **net\_params**, which contains
network parameters specified during task initialization. Unlike most
other parameters, net\_params may vary drastically dependent on the
specific network configuration. For the ring road, the network
parameters will include a characteristic "radius", in addition other
values that will be mentioned as we move through this section. We begin
by creating a new python script titled ``my_generator.py``. In this
file, import cistar's base generator class and some mathematical
functions from ``numpy`` that we may need.

::

    # import cistar's base generator
    from cistar_dev.core.generator import Generator

    # some mathematical operations that may be used
    from numpy import pi, sin, cos, linspace

Afterwards, we begin creating a class titled ``myGenerator`` that
inherits the properties of cistar's base generator.

::

    # define the generator class, and inherit properties from the base generator
    class myGenerator(Generator):

Once the base generator has been inherited, creating a child class
becomes very systematic. All child classes are required to define at
least the following three function: ``specify_nodes``,
``specify_edges``, and ``specify_routes``. In addition, the following
optional functions also may be specified: ``specify_types``,
``specify_connections``, ``specify_rerouters``. All of the functions
mentioned in the above paragraph take in as input net\_params, and
output a list of dictionary elements, with each element providing the
attributes of the component to be specified.

2.2 Defining the Location of Nodes
----------------------------------

The nodes of a network are the positions of a select few points in the
network. These points are connecting together using edges (see section
2.3), and these edges serves as the lanes the make up the traffic
network. For the ring network, we place four nodes, as seen in figure
([STRIKEOUT:blank]). ([STRIKEOUT:add figure here]) In order to specify
the location of the nodes that will be placed in the network, the
function ``specify_nodes`` is used. This function provides the base
class with a list of dictionary elements, with the elements containing
attributes of the nodes. These attributes must include:

-  **id**: name of the node
-  **x**: x coordinate of the node
-  **y**: y coordinate of the node

Other possible attributes may be found at:
http://sumo.dlr.de/wiki/Networks/Building_Networks_from_own_XML-descriptions#Node_Descriptions
In order to properly specify the nodes, add the follow function to the
generator class:

::

    def specify_nodes(self, net_params):
        # one of the elements net_params will need is a "radius" value
        r = net_params["radius"]

        # specify the name and position (x,y) of each node
        nodes = [{"id": "bottom", "x": repr(0),  "y": repr(-r)},
                 {"id": "right",  "x": repr(r),  "y": repr(0)},
                 {"id": "top",    "x": repr(0),  "y": repr(r)},
                 {"id": "left",   "x": repr(-r), "y": repr(0)}]

        return nodes

2.3 Defining the Properties of Edges
------------------------------------

Once the nodes are specified, directed edges are needed to connect these
nodes together. The attributes of these edges are defined in the
``specify_edges`` function, and must include:

-  **id**: name of the edge
-  **from**: name of the node the edge starts from
-  **to**: the name of the node the edges ends at
-  **length**: length of the edge

In addition, the attributes must contain at least one of the following:

-  **numLanes** and **speed**: the number of lanes and speed limit of
   the edge, respectively

or

-  **type**: a type identifier for several edges, which can be used if
   several lanes possess similar properties (number of lanes, speed
   limit, etc...)

The inclusion of a type necessitates the addition of a specify\_types
function, which is outside the scope of this tutorial; accordingly, the
former option will be used. Other possible attributes can be found at:
http://sumo.dlr.de/wiki/Networks/Building_Networks_from_own_XML-descriptions#Edge_Descriptions.
One useful attribute is **shape**, which specifies the shape of the edge
connecting the two nodes. The shape consists of a series of subnodes
(internal to sumo) that are connected together by straight lines to
create a shape. If no shape is specified, the nodes are connected by a
straight line. This attribute will be needed to create the circular arcs
between the nodes in the system. In order to properly specify the edges
of the ring road, add the follow function to the generator class:

::

    def specify_edges(self, net_params):
        r = net_params["radius"]
        edgelen = r * pi / 2
        # the resolution specifies the number of subnodes making up the circular arcs
        resolution = net_params["resolution"]
        # this will let us control the number of lanes in the network
        lanes = net_params["lanes"]
        # speed limit of vehicles in the network
        speed_limit = net_params["speed_limit"]

        edges = [{"id": "bottom", "numLanes": repr(lanes), "speed": repr(speed_limit),
                  "from": "bottom", "to": "right", "length": repr(edgelen),
                  "shape": " ".join(["%.2f,%.2f" % (r * cos(t), r * sin(t))
                                     for t in linspace(-pi / 2, 0, resolution)])},
                 {"id": "right", "numLanes": repr(lanes), "speed": repr(speed_limit),
                  "from": "right", "to": "top", "length": repr(edgelen),
                  "shape": " ".join(["%.2f,%.2f" % (r * cos(t), r * sin(t))
                                     for t in linspace(0, pi / 2, resolution)])},
                 {"id": "top", "numLanes": repr(lanes), "speed": repr(speed_limit),
                  "from": "top", "to": "left", "length": repr(edgelen),
                  "shape": " ".join(["%.2f,%.2f" % (r * cos(t), r * sin(t))
                                     for t in linspace(pi / 2, pi, resolution)])},
                 {"id": "left", "numLanes": repr(lanes), "speed": repr(speed_limit),
                  "from": "left", "to": "bottom", "length": repr(edgelen),
                  "shape": " ".join(["%.2f,%.2f" % (r * cos(t), r * sin(t))
                                     for t in linspace(pi, 3 * pi / 2, resolution)])}]

        return edges

2.4 Defining Routes Vehicles can Take
-------------------------------------

The routes are the sequence of edges vehicles traverse given their
current position. For example, the vehicle in figure ([STRIKEOUT:blank])
begins in the edge titled "bottom", and from there must follow traverse,
in sequence, the edges "bottom", "right", top", and finally, "left",
before restarting its path. In order to specify the routes given all
possible starting edges, the function ``specify_routes`` is used. This
function outputs a single dict file, in which the keys are the names of
all starting edges, and the items are the sequence of edges the vehicle
must follow starting from the current edge, as described in the previous
paragraph. Taking into consideration that vehicles may begin in any of
the four edges, the routing function is defined as follows:

::

    def specify_routes(self, net_params):
        rts = {"top":    ["top", "left", "bottom", "right"],
               "left":   ["left", "bottom", "right", "top"],
               "bottom": ["bottom", "right", "top", "left"],
               "right":  ["right", "top", "left", "bottom"]}

        return rts

2.5 Adding Rerouters for Continuous Movement
--------------------------------------------

In order to ensure that vehicles continue to traverse the network after
their initial pass, rerouters are added to sumo using the function
``specify_rerouters``. The function outputs a list of dictionaries, with
each dictionary containing three elements:

-  **name**: name of the rerouter
-  **from**: the edge in which rerouting takes place
-  **route:** name of the route the vehicle is rerouted into

In order to add rerouters to the configuration files, add the following
function to the generator:

::

    def specify_rerouters(self, net_params):
        rerouting = [{"name": "rerouterTop",    "from": "top",    "route": "routebottom"},
                     {"name": "rerouterBottom", "from": "bottom", "route": "routetop"},
                     {"name": "rerouterLeft",   "from": "left",   "route": "routeright"},
                     {"name": "rerouterRight",  "from": "right",  "route": "routeleft"}]

        return rerouting

3 Creating a Scenario
=====================

This section walks you through creating a custom scenario class. The
scenario class is used to specify the locations of edges in the network,
as well as the positions of vehicles at the start of a rollout.

3.1 Inheriting the Base Scenario Class
--------------------------------------

Similar to the generator we created in section 2, we begin by inheriting
the methods from the base scenario class located in
``cistar_dev/core/scenarios.py``. This base class takes as inputs a few
variables, some of which will will be discussed in section 5. From a
high level, the inputs are as follows:

-  **name** (string): the name assigned to the scenario
-  **generator\_class** (generator type): the generator class we created
   in section 2
-  **type\_params** (list of tuples): types and number of vehicles to be
   placed in the network
-  **net\_params** (dict): network configuration parameters, fairly
   unique to the network being created
-  **cfg\_params** (dict): a few configuration parameters, such as the
   path where the configuration files will be placed
-  **initial\_config** (dict): parameters used to modify the initial
   positions of vehicles in the network

In order to begin creating your scenario class, create a new script in
the same folder as your generator class titled ``my_scenario.py``. Begin
this script by importing cistar's base scenario class.

::

    # import cistar's base scenario class
    from cistar_dev.core.scenario import Scenario

Then, create your scenario titled ``myScenario`` with the base scenario
class as its parent.

::

    # define the scenario class, and inherit properties from the base scenario class
    class myScenario(Scenario):

3.2 Specifying the Length of the Network (optional)
---------------------------------------------------

The base scenario class will look for a "length" parameter in
net\_params upon initialization. However, this value is implicitly
defined by the radius of the ring, making specifying the length a
redundancy. In order to avoid any confusion when creating ``net_params``
during an experiment run (see sections 5 and 6), the length of the
network can be added to ``net_params`` via scenario subclass's
initializer. This is done by defining the initializer as follows:

::

    from numpy import pi

    def __init__(self, name, generator_class type_params, net_params, cfg_params=None,
                 initial_config=None):
        # add to net_params a characteristic length
        net_params["length"] = 4 * pi * net_params["radius"]

Then, the initializer is finished off by adding the base (super) class's
initializer:

::

        super().__init__(name, generator_class, type_params, net_params, cfg_params, initial_config)

3.3 Specifying the Starting Position of Edges
---------------------------------------------

The starting position of the edges are the only adjustments to the
scenario class that *need* to be performed in order to have a fully
functional subclass. These values specify the distance the edges within
the network are from some reference, in one dimension. To this end, up
to three functions may need to be overloaded within the subclass:

::

    specify_edge_starts: defines edge starts for road sections with respect to some global reference
    specify_intersection_edge_starts (optional): defines edge starts for intersections with respect to some global reference frame. Does note need to be specified if no intersections exist.
    specify_internal_edge_starts: defines the edge starts for internal edge nodes (caused by finite length connections between road section)

All of the above mentioned function receive no inputs and output a list
of tuple, in which the first element of the tuple is the name of the
edge/intersection/internal\_link, and the second value is the distance
of the component from some global reference, i.e.
``[(component_0, pos_0, component_1, pos_1, ...]``. In section 2, we
created a network with 4 edges named: "bottom", "right", "top", and
"left", each with starting nodes with the same as the edge. We will
assume that the node titled "bottom" is the origin, and accordingly the
position of the edge start of edge "bottom" is ``0``. The edge called
"right", on the other hand, starts at node "right", which is a quarter
of the length of the network from the node "bottom", and accordingly the
position of its edge start is ``radius * pi/2``. This process continues
for each of the edges. We can then define the starting position of the
edges as follows:

::

    def specify_edge_starts(self):
        r = net_params["radius"]

        edgestarts = [("bottom", 0),
                      ("right", r * 1/2 *pi),
                      ("top", r * pi),
                      ("left", r * 3/2 * pi)]

        return edgestarts

Our road network does not contain intersections, and internal links are
not used in this experiment and outside the scope of the problem.
Accordingly, the methods ``specify_intersection_edge_starts`` and
``specify_internal_edge_starts`` are not used in this example.

3.4 Controlling the Starting Positions of Vehicles
--------------------------------------------------

Cistar v0.1 supports the use of several positioning methods for closed
network systems. As can be seen in figure ([STRIKEOUT:blank]), these
methods include:

-  a **uniform** distribution, in which all vehicles are placed
   uniformly spaced across the length of the network,
-  an **upstream** distribution, in which vehicles are placed in the
   network with exponentially increasing headways,
-  a **gaussian** distribution, in which the vehicles are perturbed from
   this uniform starting position following a gaussian,
-  and a **gaussian-additive** distribution, in which vehicle are placed
   sequentially following a gaussian distribution, thereby causing the
   error to build up

([STRIKEOUT:add figures here]) In addition to the above distributions,
the user may specify a custom set of starting position by overriding the
function ``gen_custom_start_pos``. This is not part of the scope of this
tutorial, and will not be covered.

4 Creating an Environment
=========================

The environment class is the primary functioning component after the
network is initialzed. This class ties the components of SUMO and
``rllab`` together, running a system of vehicles in a network over
discrete time steps, while treating some of these vehicles as
reinforcement learning agents whose actions are specified by ``rllab``.

4.1 Inheriting the Base Environment Class
-----------------------------------------

For the third and final time, we will begin by inheriting a core base
class from cistar. The core environment class is located in
``cistar_dev/core/base_env.py`` . This class contains the bulk of the
SUMO-related operations needed during a run, such as specifying actions
to be performed by vehicles and collecting information on the
network/vehicles for any given time step. In addition, the base
environment accepts states, actions, and reward values and provides them
to the reinforcement learning algorithm in ``rllab``, which then trains
the reinforcement learning agent(s) (i.e. the autonomous vehicles) to
maximize their reward.

Begin by creating a new script in the directory ``cistar_dev/envs``
titled ``my_environment.py``. Begin this script by importing cistar's
base environment class.

::

    # import the base environment class
    from cistar_dev.core.base_env import SumoEnvironment

In addition to cistar's base environment, we will import a few methods
from ``gym``, which will allow the environment to be compatible with the
requirements of a Gym Environment (see section 6.1). The first method we
will need is ``Box``, we is used to define a bounded array of values in

.. math:: \mathbb{R}^n

. The second method we will import is ``Tuple``, we allows us to combine
multiple ``Box`` elements together. In order to import these terms, add
the following lines to your script.

::

    from gym.spaces.box import Box
    from gym.spaces.tuple_space import Tuple

Now, create your environment class titled ``myEnvironment`` with the
base environment class as its parent.

::

    # define the environment class, and inherit properties from the base environment class
    class myEnvironment(SumoEnvironment):

By inheriting cistar's base environment, a proper reinforcement learning
environment can be created by adding the following functions to the
child class: ``action_space``, ``observation_space``,
``apply_rl_action``, ``get_state``, and ``compute_reward``, which are
discussed in the next few subsections.

4.2 Specifying an Action Space
------------------------------

The action space of an environment represents the number of actions a
given reinforcement learning agent can perform and the bounds of those
actions. Autonomous vehicles may perform several different actions, such
as modifying their accelerations and lane-changing to the lanes on their
sides; however, in a single-lane ring road setting, vehicles can only
reasonably perform accelerations. Moreover, these accelerations are
bounded by maximum and minimum acceleration values it can reasonably
achieve.

The components of the action space are in the function conveniently
called ``action_space``; accordingly, we begin by defining this
function:

::

    @property
    def action_space(self):

The above function does not take as input any values; however, it is
part of the environment class, and accordingly has access to all of its
attributes. One such attribute of interest is the number of autonomous
vehicles in the system, located under ``self.scenario.num_rl_vehicles``.
Another significant attribute is the variable ``self.env_params``, which
contains several evironment-specific parameters. Two such parameters in
this case are the maximum possible accelerations and decelerations of
the reinforcement learning agents, under the key "max-acc" and
"max-deacc", respectively . Given these attributes, we may specify the
number actions performed by the rl agent and bounds of these actions as
follows:

::

        num_acc_actions = self.scenario.num_rl_vehicles
        acc_upper_bound = self.env_params["max-acc"]
        acc_lower_bound = - abs(self.env_params["max-deacc"])

Once the parameters of the action space are specified, the Box element
discussed early in section 4.2 may be filled as follows:

::

       acc_action_space = Box(low=acc_lower_bound, high=acc_upper_bound, shape=num_acc_actions)

       return acc_action_space

4.3 Specifying an Observation Space
-----------------------------------

The observation space of an environment represents the number and types
of observations that are provided to the reinforcement learning agent.
For a network of vehicles in a single lane setting, the observation
space consists of a vector of velocities

.. math:: v

and absolute positions

.. math:: x

for each vehicle in the network.

We begin by defining our ``observation_space`` function:

::

    @property
    def observation_space(self):

In this function, we create two Box elements; one for the absolute
positions of the vehicles, and another for the speeds of the vehicles.
These values may range from zero to infinity, and there are
``self.scenario.num_vehicles`` number of unique values for each of them:

::

        speed = Box(low=0, high=np.inf, shape=(self.scenario.num_vehicles,))
        absolute_pos = Box(low=0., high=np.inf, shape=(self.scenario.num_vehicles,))

Finally, we combine the two ``Box`` elements using the Tuple method.
This tuple used at the output from the ``observation_space`` function:

::

        return Tuple([speed, absolute_pos])

4.4 Applying Actions to the Autonomous Vehicles
-----------------------------------------------

The function ``apply_rl_action`` acts as the bridge between rllab and
sumo, transforming commands specified by rllab in the action space into
actual action in the traffic scenario created within sumo. This function
takes as an input the actions requested by rllab, and sends the commands
to SUMO without returning any output. We begin by defining it:

::

    def apply_rl_actions(self, rl_actions):

Taking into consideration the action space specified in section 4.2, the
array of rl actions provided to ``apply_rl_action`` consists solely of
the accelerations the autonomous vehicles need to perform. These values
may be turned into accelerations in SUMO using the function
``apply_acceleration`` , which takes as inputs a list of vehicle
identifiers and acceleration values, and sends the proper commands to
SUMO. Using this function, the method needed to apply rl actions is
simply as follows:

::

        rl_ids = self.rl_ids  # the variable self.rl_ids contains a list of the names of all rl vehicles
        self.apply_acceleration(rl_ids, rl_actions)

4.5 Collecting the State Space Information
------------------------------------------

As mentioned in section 4.3, the observation space consists of the speed
and position of all vehicles in the network. In order to supply the rl
algorithm with these values, the function ``get_state`` is used. This
function returns a matrix containing the components of the observation
space to the base environment.

In order to collect the states of specific vehicles in the network for
the current time step, the variable ``self.vehicles`` can be used. This
variable consists of a dictionary with key elements equal to the vehicle
ids (listed in ``self.ids``). The elements of the ``self.vehicles`` are
also dictionaries, with the keys denoting the state being stored, such
as "speed", "edge", "absoluteposition", etc... In order to create the
necessary matrix of states, the function get\_state loops through the
vehicle ids of all vehicles in the network, and collects for each
vehicle its speed and absolute position:

::

        state = np.array([[self.vehicles[veh_id]["speed"], self.vehicles[veh_id]["absolute_position"]]
                          for veh_id in self.sorted_ids])

        return state

4.6 Computing an Appropriate Reward Function
--------------------------------------------

The reward function is the component which the reinforcement learning
algorithm will attempt to maximum over. This is defined in the function
``compute_reward``:

::

    def compute_reward(self, state, rl_actions, **kwargs):

We choose a simple reward function to encourage high system-level
velocity. This function measures the deviation of a system of vehicles
from a user-specified desired velocity, peaking when all vehicles in the
ring are set to this desired velocity. Moreover, in order to ensure that
the reward function naturally punishing the early termination of
rollouts due to collisions or other failures, the function is formulated
as a mapping

.. math:: r : S\times A \to R \geq 0

. This is done by subtracting the deviation of the system from the
desired velocity from the peak allowable deviation from the desired
velocity. Additionally, since the velocity of vehicles are unbounded
above, the reward is bounded below by zero, to ensure nonnegativity.

Define

.. math:: v_{des}

as the desired velocity,

.. math:: 1^k

 a vector of ones of length :math:`k`

.. math:: k

,

.. math:: n

 as the number of vehicles in the system, and

.. math:: v

 as a vector of velocities. The reward function is formulated as:

.. math:: r(v) = \max{0, ||v_{des} \cdot 1^k ||_2 - || v - v_{des} \cdot 1^k ||_2}

**4.6.1 Using Built-in Reward Functions** Cistar come with several
built-in reward functions located in ``cistar_dev.core.rewards`` and
``cistar_dev.core.multi_agent_rewards``. In order to use these reward
function, we begin by importing these reward function at the top of the
script:

::

    # cistar's built-in reward functions
    from cistar_dev.core import rewards
    from cistar_dev.core import multi_agent_rewards

One reward function located in the ``rewards`` file is the function
``desired_velocity``, which computes the reward described in this
section. It takes as input the environment variable (``self``) and a
"fail" variables that specifies if the vehicles in the network
experiences any sort of crash, and is an element of the ``**kwargs``
variable. Returning to the ``compute_reward`` function, the reward may
be specified as follows:

::

        return rewards.desired_velocity(self, fail=kwargs["fail"])

**4.6.2 Building the Reward Function** In addition to using cistar's
built-in reward functions, you may also choose to create your own
functions from scratch. In doing so, you may choose to use as inputs the
state, actions, or environment (self) variables, as they are presented
in the current time step. In addition, you may use any available
``**kwargs`` variables. In the most general setting, ``kwargs`` will
come with a "fail" element, which describes whether a crash or some
other failure has occurred within the network. In order to prevent the
reward function from outputting a reward when a fail has occurred, we
begin by setting all rewards to zero when "fail" is true:

::

        if kwargs["fail"]:
            return 0

Next, we collect the cost of deviating from the desired velocity. This
is done by taking the two-norm of the difference between the current
velocities of vehicles and their desired velocities.

::

        vel = np.array([self.vehicles[veh_id]["speed"] for veh_id in self.ids])

        cost = vel - self.env_params["target_velocity"]
        cost = np.linalg.norm(cost)

Finally, in order to ensure the value remains positive, we subtract this
deviation from the maximum allowable deviation, and clip the value from
below by zero.

::

        max_cost = np.array([self.env_params["target_velocity"]] * len(self.ids))
        max_cost = np.linalg.norm(max_cost)

        return max(max_cost - cost, 0)

4.7 Registering the Environment as a Gym Environment
----------------------------------------------------

In order to run reinforcement learning experiments (see section 6), the
experiment we created needs to be registered as a Gym Environment. In
order for cistar to register your environment as a Gym Environment, go
to ``cistar_dev/envs/__init__.py``, and add the following line:

::

    from cistar_dev.envs.my_environment import myEnvironment

5. Running an Experiment without Autonomy
=========================================

Once the classes described in sections 2, 3, and 4 are created, we are
now ready to run experiments with cistar. We begin by running an
experiment without any learning/autonomous agents. This experiment acts
as our control case, and helps us ensure that the system exhibits the
sorts of performance deficiencies we expect to witness. In the case of a
single-lane ring road, this deficiency is the phenomenon known as string
instability, in which vehicles begin producing stop-and-go waves among
themselves ([STRIKEOUT:reference Sugiyama et. al]).

5.1 Importing the Necessary Modules
-----------------------------------

In order to run the experiment in the absence of autonomy, we will
create a ``SumoExperiment`` variable. This variable takes as input the
environment and scenario classes developed in sections 3 and 4. Note
that the generator class is not needed by the experiment class, but
rather by the scenario class.

We begin by creating a new script in the same directory as that of the
generator and scenario classes titled ``my_control_experiment.py``. In
this script, we import the base experiment class, as well as the
generator, scenario, and environment subclasses we developed.

::

    # this is the base experiment class
    from cistar_dev.core.exp import SumoExperiment

    # these are the classes I created
    from ./my_generator import myGenerator
    from ./my_scenario import myScenario
    from cistar_dev/envs/my_environment import myEnvironment

    # for possible mathematical operation we may want to perform
    import numpy as np

In order to impose realistic vehicle dynamics on the vehicles in the
network, cistar possesses a few acceleration and lane-changing
controller classes. These classes are imported into the script as
follows:

::

    from cistar_dev.controllers.car_following_models import *
    from cistar_dev.controllers.lane_change_controllers import *

5.2 Setting Up the Environment and Scenario Classes
---------------------------------------------------

In order to initialize scenario and environment classes (as well as the
generator class which is initialized within the scenario), the inputs
for each class, must be must be specified. These inputs are:
``sumo_params``, ``sumo_binary``, ``type_params``, ``env_params``,
``net_params``, ``cfg_params``, and (optionally) ``initial_config``.
``sumo_params`` is used to pass the time step and sumo-specified safety
modes, which constrain the dynamics of vehicles in the network to
prevent crashes. We will use this parameter to specify a step size of a
0.1 s.

::

    sumo_params = {"time_step": 0.1}

sumo\_binary allows us to specify whether we would like see sumo's gui
during the experiment's runtime. If you would like to see the gui, set
this term to "sumo-gui"; otherwise, set it to "sumo". For our first
experiment, we would like to see the gui:

::

    sumo_binary = "sumo-gui"

``type_params`` is used to specify the types of vehicles in the network.
This variable consists of a list of tuples, with each tuple containing
five elements:

-  first element (string): some identifier for the specific type. Each
   vehicle of this type will have an id beginning with this string.
-  second element (int): the number of vehicles of this type
-  third element (tuple): used to specify the acceleration dynamics of
   the vehicles. The first component of the tuple is an acceleration
   controller class provided by
   ``cistar_dev.controllers.car_following_models``, while the second
   component is a dict that optionally allows you to control the
   coefficients of the acceleration model, but may be left empty.
-  fourth element (tuple *or* None type): used to specify the
   lane-changing dynamics of the vehicles. If a tuple is provided, then
   the first component is a lane-changing controller class provided by
   ``cistar_dev.controllers.lane_change_controllers``, while the second
   component is a dict that optionally allows you to control the
   coefficients of the lane-changing model, but may be left empty. If a
   None value is provided, then SUMO dictates the lane-changing behavior
   of vehicles.
-  fifth element (float): initial velocity of the vehicles, in m/s

For this experiment, we would like to place 22 vehicles in the ring that
follow the acceleration dynamics described by the Intelligent Driver
Model ([STRIKEOUT:reference]).

Accordingly, the ``type_params`` variable is defined as follows:

::

    type_params = [("human", 22 - 1, (IDMController, {}), None, 0)]

``env_params`` provides several environment and experiment-specific
parameters. This includes specifying the parameters of the action space
and relevant coefficients to the reward function. Whether autonomous
vehicles are placed within the network or not, the environment will
attempt to create an action space. Accordingly, we provide
``env_params`` with the necessary components for this method:

::

    env_params = {"max-acc": 3, "max-deacc": -6}

``net_params`` consist of a dictionary of several network-specific
values of interest. Given the generator class we created in section 2,
these values include: "radius", "lanes", "speed\_limit", and
"resolution". In addition, a "net\_path" component is used to specify
where the network xml files created by the generator class will be
placed.

::

    net_params = {"radius": 230/(2*np.pi), "lanes": 1, "speed_limit": 30, "resolution": 40, "net_path": "debug/net/"}

Note that, if section 3.2 was not implemented when creating the scenario
class, an additional "length" component must be added to ``net_params``
as follows:

::

    net_params["length"] = net_params["radius"] * 2 * np.pi

``cfg_params`` is used to specify a few configuration parameters, such
as the start time of a run and the location where the configuration
files developed by the generator should be placed. This variable is
defined as follows:

::

    cfg_params = {"start_time": 0, "cfg_path": "debug/cfg/"}

Finally, the variable ``initial_config`` affects the positioning of
vehicle in the network at the start of a rollout. In order to prevent
the system from being perfectly symmetric, we set the "spacing" of the
vehicles in the network to be "gaussian\_additive" (see section 3.4):

::

    initial_config = {"spacing": "gaussian_additive"}

Once all the necessary inputs are prepared, the scenario and environment
variables can be initialized. Moreover, naming the experiment
"ring\_road\_all\_human", the classes are created as followed:

::

    # creating a scenario variable
    scenario = myScenario("ring_road_all_human", myGenerator, type_params, net_params, 
                          cfg_params, initial_config)

    # creating an environment variable
    env = myEnvironment(env_params, sumo_binary, sumo_params, scenario)

5.3 Setting up the Experiment Class
-----------------------------------

Once the environment and scenario classes are ready, the experiment
variable can be creating as follows:

::

    # creating an experiment variable
    exp = SumoExperiment(env, scenario)

This allows us to run the experiment for as many runs and any number of
time steps we would like. In order to run the experiment for 1 run of
150 seconds, we specify the following values:

::

    num_runs = 1  # I would like to run the experiment once
    num_steps = 150 / sumo_params["time_step"]  # I would like the experiment to run for 150 sec

Finally, we get the script to run the experiment by adding the following
line:

::

    exp.run(num_runs, num_steps)

5.4 Running the Experiment
--------------------------

Now that all the necessay classes are ready and the experiment script is
prepared, we can finally run our first experiment. Run the script titled
``my_control_experiment.py`` from your IDE or from the terminal. After a
few seconds, a gui should appear on the screen with a circular road
network, as seen in figure ([STRIKEOUT:blank]) below. Click on the play
button (circled in red in figure [STRIKEOUT:blank]) and the network will
be filled with vehicles, which then begin to accelerate. ([STRIKEOUT:add
figure with gui here]) (describe what we see, show velocity and
space-time diagrams)

6. Running an Experiment with Autonomy
======================================

Finally, we will attempt to add autonomous vehicles in the ring road. We
will begin by adding a single autonomous vehicles, in hopes that this
vehicle may be able to learn to stabilize the ring.

## 6.1 Creating a Gym Environment

Unlike in section 5, we will not rely on cistar's SumoExperiment
variables to run experiments, but rather will create a ``GymEnv``
supported by ``rllab``. This will serve as the reinforcement learning
agent's digital "playground" as it tries to improve the performance of
the vehicles in the network. Create a new script entitled
``my_rl_experiment.py`` and import the generator and scenario
subclasses, in addition to the dynamical model provided by cistar, as
you had done in section 5.1 for the control experiment:

::

    # these are the classes I created
    from ./my_generator import myGenerator
    from ./my_scenario import myScenario

    # for possible mathematical operation we may want to perform
    import numpy as np

    # acceleration and lane-changing controllers for human-driven vehicles
    from cistar_dev.controllers.car_following_models import *
    from cistar_dev.controllers.lane_change_controllers import *

A new controller that is used in this experiment and needed in the case
of mixed-autonomy is the ``RLController``, located in
``cistar_dev.controllers.rlcontroller``. Any types of vehicles with this
controller will act as reinforcement learning agent(s).

::

    from cistar_dev.controllers.rlcontroller import RLController

In additon, we will need several functions from ``rllab``:

::

    from rllab.envs.normalized_env import normalize
    from rllab.misc.instrument import run_experiment_lite
    from rllab.algos.trpo import TRPO
    from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
    from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
    from rllab.envs.gym_env import GymEnv

In this script, we will define a function called ``run_task`` that will
be used to create and run our gym environment:

::

    def run_task(v):

Within this function, import the environment you created so that the Gym
Envrionment variable may be able to locate it:

::

        from ./my_environment import myEnvironment

Similar to section 5, we must now define the necessary input variables
to the generator, scenario, and environment classes. These variable will
larger remain the same but with the addition of a few component.

For one, in ``sumo_params`` we will want to specify an aggressive
SUMO-defined speed mode, which will prevent SUMO from enforcing a safe
velocity upper bound on the autonomous vehicle, but may lead to the
autonomous vehicles crashing into the vehicles ahead of them. This is
done by setting "rl\_sm" to "aggressive".

Moreover, in order to run rollouts with a max path length of 1500 steps
(i.e. 150 s), we set "num\_steps" in ``env_params`` to 1500. In
addition, in order to train the vehicle to move the vehicles in the
network as fast as possible, we set "target\_velocity" in ``env_params``
to 8 m/s (far beyond the expected equilibrium velocity).

Finally we introduce an autonomous (rl) vehicle into the network by
reducing the number of human vehicles by 1 and adding a new tuple to the
``type_params`` list for a vehicle with the acceleration controller
``RLController``.

The final set of input variables are as follows:

::

        sumo_params = {"time_step": 0.1, "rl_sm": "aggressive"}
        sumo_binary = "sumo-gui"

        env_params = {"target_velocity": 8, "max-deacc": -6, "max-acc": 3, "num_steps": 1500,}

        net_params = {"length": 230, "lanes": 1, "speed_limit": 30, "resolution": 40,
                      "net_path": "debug/net/"}

        cfg_params = {"start_time": 0, "cfg_path": "debug/rl/cfg/"}

        initial_config = {"spacing": "gaussian_additive"}

        num_cars = 22

        type_params = [("rl", 1, (RLController, {}), None, 0),
                       ("human", num_cars - 1, (IDMController, {}), None, 0)]

Creating the scenario does not change between this section and the last.
Calling our scenario "stabilizing-the-ring", the scenario class is
initialized as follows:

::

        scenario = myScenario("stabilizing-the-ring", myGenerator, type_params, net_params,
                              cfg_params, initial_config)

The environment, however, is no longer defined in the same manner.
Instead, a variable called env\_name is specified with the name of the
environment you created, and the list of parameters are placed into a
tuple:

::

        env_name = "myEnvironment"
        pass_params = (env_name, sumo_params, sumo_binary, type_params, env_params, net_params,
                       cfg_params, initial_config, scenario)

Then, the Gym Environment is initialized as follows:

::

        env = GymEnv(env_name, record_video=False, register_params=pass_params)

6.2 Specifying the Necessary rllab Components
---------------------------------------------

We use linear feature baselines and Trust Region Policy Optimization
([STRIKEOUT:reference]) for learning the policy, with discount factor

.. math:: \gamma = 0.999

, and step size 0.01. A diagonal Gaussian MLP policy is used with hidden
layers (100, 50, 25) and tanh non-linearity. This is done within your
script by adding the following lines of code to the ``run_task``
function:

::

        horizon = env.horizon
        env = normalize(env)

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(100, 50, 25)
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=15000,
            max_path_length=env.horizon,
            n_itr=300,
            # whole_paths=True,
            discount=0.999,
        )
        algo.train(),

6.3 Setting up the Experiment
-----------------------------

Once the function run\_task is complete, we are able to wrap up the
script by calling ``rllab`` to run the experiment. This is done through
the use of the ``run_experiment_lite`` function. We choose to run the
experiment locally with one worker for sampling and a seed value of 5.
Also, we would like to keep track of the policy parameters from all
iterations.

::

    run_experiment_lite(
        run_task,
        # Number of parallel workers for sampling
        n_parallel=1,
        # Keeps the snapshot parameters for all iterations
        snapshot_mode="all",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=seed,
        mode="local",
        exp_prefix="stabilizing-the-ring",
    )

Note that, for Linux builds, it may be necessary to specify the path to
the location of ``rllab``'s python command within
``run_experiment_lite`` . This will look something similar to:

::

        python_command="<acaconda2_directory>/envs/rllab-distributed/bin/python3.5"

6.4 Visualizing Rollouts
------------------------
