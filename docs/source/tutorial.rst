Tutorial
******************

1. Introduction
===============

This tutorial is intended for readers who are new to Flow. While some
reinforcement learning terms are presented within the contents of this tutorial,
no prior background in the field is required to successfully complete any
steps. Be sure to install Flow before starting this tutorial.

1.1. About Flow
-----------------

Flow is a framework for deep reinforcement learning in
mixed-autonomy traffic scenarios. It interfaces the RL library ``rllab``
with the traffic microsimulator ``SUMO``. Through Flow, autonomous
vehicles may be trained to perform various tasks that improve the
performance of traffic networks. Currently, Flow v0.1 supports the
implementation of simple closed networks, such as ring roads, figure
eights, etc... In order to run an experiment on Flow, three objects are
required:

-  A **Generator**: Generates the configuration files needed to create
   a transportation network with ``SUMO``.
-  A **Scenario**: Specifies the location of edge nodes in the network,
   as well as the positioning of vehicles at the start of a run.
-  An **Environment**: ties the components of ``SUMO`` and ``rllab`` together,
   running a system of vehicles in a network over discrete time steps,
   while treating some of these vehicles as reinforcement learning
   agents whose actions are specified by ``rllab``.

Once the above classes are ready, an **experiment** may be prepared to
run the environment for various levels of autonomous vehicle penetration
ranging from 0% to 100%.


1.2. About this tutorial
------------------------

In this tutorial, we will create a simple ring road network, which in the
absence of autonomous vehicles, experience a phenomena known as "stop-and-go
waves". An autonomous vehicle in then included into the network and trained
to attenuate these waves. The remainder of the tutorial is organized as follows:

-  In Sections 2, 3, and 4, we create the primary classes needed to run
   a ring road experiment.
-  In Section 5, we run an experiment in the absence of autonomous
   vehicles, and witness the performance of the network.
-  In Section 6, we run the experiment with the inclusion of autonomous
   vehicles, and discuss the changes that take place once the
   reinforcement learning algorithm has converged.


.. _creating-a-generator:

2 Creating a Generator
======================

This section walks you through the steps needed to create a generator class.
The generator prepares the configuration files needed to create a
transportation network in sumo. A transportation network can be thought
of as a directed graph consisting of nodes, edges, routes, and other
(optional) elements.

.. _inheriting-the-base-generator:

2.1 Inheriting the Base Generator
---------------------------------

We begin by creating a file called ``my_generator.py``. In this file, we
create a class titled ``myGenerator`` that inherits the properties of Flow's
base generator class.

::

    # import Flow's base generator
    from flow.core.generator import Generator

    # some mathematical operations that may be used
    from numpy import pi, sin, cos, linspace

    # define the generator class, and inherit properties from the base generator
    class myGenerator(Generator):

The base generator class accepts a single input:

* **net\_params**: contains network parameters specified during task
  initialization. Unlike most other parameters, net\_params may vary drastically
  dependent on the specific network configuration. For the ring road, the
  network parameters will include a characteristic radius, number of lanes,
  and speed limit.

Once the base generator has been inherited, creating a child class
becomes very systematic. All child classes are required to define at
least the following three function: ``specify_nodes``,
``specify_edges``, and ``specify_routes``. In addition, the following
optional functions also may be specified: ``specify_types``,
``specify_connections``, ``specify_rerouters``. All of the functions
mentioned in the above paragraph take in as input net\_params, and
output a list of dictionary elements, with each element providing the
attributes of the component to be specified.

.. _defining-the-location-of-nodes:

2.2 Defining the Location of Nodes
----------------------------------

The nodes of a network are the positions of a select few points in the
network. These points are connected together using edges (see `section
2.3`_). For the ring network, we place four nodes at the bottom, right, left,
and right of the ring.

.. _section 2.3: defining-the-properties-of-edges_

In order to specify the location of the nodes that will be placed in the
network, the function ``specify_nodes`` is used. This function provides the
base class with a list of dictionary elements, with the elements containing
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
        r = net_params.additional_params["radius"]

        # specify the name and position (x,y) of each node
        nodes = [{"id": "bottom", "x": repr(0),  "y": repr(-r)},
                 {"id": "right",  "x": repr(r),  "y": repr(0)},
                 {"id": "top",    "x": repr(0),  "y": repr(r)},
                 {"id": "left",   "x": repr(-r), "y": repr(0)}]

        return nodes

.. _defining-the-properties-of-edges:

2.3 Defining the Properties of Edges
------------------------------------

Once the nodes are specified, the nodes are linked together using directed
edges. The attributes of these edges are defined in the ``specify_edges``
function, and must include:

-  **id**: name of the edge
-  **from**: name of the node the edge starts from
-  **to**: the name of the node the edges ends at
-  **length**: length of the edge
-  **numLanes**: the number of lanes on the edge
-  **speed**: the speed limit for vehicles on the edge

Other possible attributes can be found at:
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
        r = net_params.additional_params["radius"]
        edgelen = r * pi / 2
        # this will let us control the number of lanes in the network
        lanes = net_params.additional_params["lanes"]
        # speed limit of vehicles in the network
        speed_limit = net_params.additional_params["speed_limit"]

        edges = [{"id": "bottom", "numLanes": repr(lanes), "speed": repr(speed_limit),
                  "from": "bottom", "to": "right", "length": repr(edgelen),
                  "shape": " ".join(["%.2f,%.2f" % (r * cos(t), r * sin(t))
                                     for t in linspace(-pi / 2, 0, 40)])},
                 {"id": "right", "numLanes": repr(lanes), "speed": repr(speed_limit),
                  "from": "right", "to": "top", "length": repr(edgelen),
                  "shape": " ".join(["%.2f,%.2f" % (r * cos(t), r * sin(t))
                                     for t in linspace(0, pi / 2, 40)])},
                 {"id": "top", "numLanes": repr(lanes), "speed": repr(speed_limit),
                  "from": "top", "to": "left", "length": repr(edgelen),
                  "shape": " ".join(["%.2f,%.2f" % (r * cos(t), r * sin(t))
                                     for t in linspace(pi / 2, pi, 40)])},
                 {"id": "left", "numLanes": repr(lanes), "speed": repr(speed_limit),
                  "from": "left", "to": "bottom", "length": repr(edgelen),
                  "shape": " ".join(["%.2f,%.2f" % (r * cos(t), r * sin(t))
                                     for t in linspace(pi, 3 * pi / 2, 40)])}]

        return edges

2.4 Defining Routes Vehicles can Take
-------------------------------------

The routes are the sequence of edges vehicles traverse given their
current position. For example, a vehicle beginning in the edge titled "bottom"
(see section 2.3) must traverse, in sequence, the edges "bottom", "right", top",
and "left", before restarting its path.

In order to specify the routes a vehicle may take, the function
``specify_routes`` is used. This function outputs a single dict element, in which
the keys are the names of all starting edges, and the items are the sequence of
edges the vehicle must follow starting from the current edge. For this network,
the available routes are defined as follows:

::

    def specify_routes(self, net_params):
        rts = {"top":    ["top", "left", "bottom", "right"],
               "left":   ["left", "bottom", "right", "top"],
               "bottom": ["bottom", "right", "top", "left"],
               "right":  ["right", "top", "left", "bottom"]}

        return rts

.. _creating-a-scenario:

3 Creating a Scenario
=====================

This section walks you through the steps required to create a scenario class.
This class is used to generate starting positions for vehicles in the
network, as well as specify the location of edges relative to some reference.

.. _inheriting-the-base-scenario-class:

3.1 Inheriting the Base Scenario Class
--------------------------------------

Similar to the generator we created in section 2, we begin by inheriting the
methods from Flow's base scenario class. Create a new script called
``my_scenario.py`` and begin the script as follows:

::

    # import Flow's base scenario class
    from flow.scenarios.base_scenario import Scenario

    # import some math functions we may use
    from numpy import pi

    # define the scenario class, and inherit properties from the base scenario class
    class myScenario(Scenario):


The inputs to Flow's base scenario class are:

-  **name**: the name assigned to the scenario
-  **generator\_class**: the generator class we created
   in `section 2`_
-  **vehicles**: used to initialize a set of vehicles in the network.
   In addition, this object contains information on the state of the vehicles
   in the network for each time step, which can be accessed during an experiment
   through various "get" functions
-  **net\_params**: see `section 2.1`_
-  **initial\_config**: affects the positioning of vehicle in the network at
   the start of a rollout. By default, vehicles are uniformly distributed in
   the network.

.. _section 2.1: inheriting-the-base-generator_

.. _section 3.2:

3.2 Specifying the Length of the Network (optional)
---------------------------------------------------

The base scenario class will look for a "length" parameter in
net\_params upon initialization. However, this value is implicitly
defined by the radius of the ring, making specifying the length a
redundancy. In order to avoid any confusion when creating net_params
during an experiment run (see sections 5 and 6), the length of the
network can be added to net_params via our scenario subclass's
initializer. This is done by defining the initializer as follows:

::

    def __init__(self, name, generator_class, vehicles, net_params,
                 initial_config=None):
        # add to net_params a characteristic length
        net_params.additional_params["length"] = 4 * pi * net_params.additional_params["radius"]

Then, the initializer is finished off by adding the base (super) class's
initializer:

::

        super().__init__(name, generator_class, vehicles, net_params, initial_config)

3.3 Specifying the Starting Position of Edges
---------------------------------------------

The starting position of the edges are the only adjustments to the
scenario class that *need* to be performed in order to have a fully
functional subclass. These values specify the distance the edges within
the network are from some reference, in one dimension. To this end, up
to three functions may need to be overloaded within the subclass:

- ``specify_edge_starts``: defines edge starts for road sections with respect
  to some global reference
- ``specify_intersection_edge_starts`` (optional): defines edge starts for
  intersections with respect to some global reference frame. Only needed by
  environments with intersections.
- ``specify_internal_edge_starts``: defines the edge starts for internal edge
  nodes caused by finite length connections between road section

All of the above functions receive no inputs and output a list
of tuples, in which the first element of the tuple is the name of the
edge/intersection/internal\_link, and the second value is the distance
of the link from some global reference, i.e.
``[(link_0, pos_0, link_1, pos_1, ...]``.

In section 2, we created a network with 4 edges named: "bottom", "right",
"top", and "left". We assume that the node titled "bottom" is the origin, and
accordingly the position of the edge start of edge "bottom" is ``0``. The edge
begins a quarter of the length of the network from the node "bottom", and
accordingly the position of its edge start is ``radius * pi/2``. This process
continues for each of the edges. We can then define the starting position of the
edges as follows:

::

    def specify_edge_starts(self):
        r = self.net_params.additional_params["radius"]

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

Flow v0.1 supports the use of several positioning methods for closed
network systems. These methods include:

-  a **uniform** distribution, in which all vehicles are placed
   uniformly spaced across the length of the network
-  a **gaussian** distribution, in which the vehicles are perturbed from
   their uniform starting position following a gaussian distribution
-  a **gaussian-additive** distribution, in which vehicle are placed
   sequentially following a gaussian distribution, thereby causing the
   error to build up

In addition to the above distributions, the user may specify a custom set of
starting position by overriding the function ``gen_custom_start_pos``. This is
not part of the scope of this tutorial, and will not be covered.

4 Creating an Environment
=========================

This section walks you through creating an environment class.
This class is the most significant component once a
network is generated. This object ties the components of ``SUMO`` and
``rllab`` together, running a system of vehicles in a network for
discrete time steps, while treating some of these vehicles as
reinforcement learning agents whose actions are specified by ``rllab``.

4.1 Inheriting the Base Environment Class
-----------------------------------------

For the third and final time, we will begin by inheriting a core base
class from Flow. Create a new script called ``my_environment.py``, and begin
by importing Flow's base environment class.

::

    # import the base environment class
    from flow.envs.base_env import SumoEnvironment

In addition to Flow's base environment, we will import a few objects
from ``gym``, which will make our environment class compatible with ``rllab``'s
base Environment class.

The first method we will need is ``Box``, which is used to define a bounded
array of values in :math:`\mathbb{R}^n`.

::

    from gym.spaces.box import Box

In addition, we will import ``Tuple``, which allows us to combine
multiple ``Box`` elements together.

::

    from gym.spaces.tuple_space import Tuple

Now, create your environment class titled ``myEnvironment`` with the
base environment class as its parent.

::

    # define the environment class, and inherit properties from the base environment class
    class myEnvironment(SumoEnvironment):

Flow's base environment class contains the bulk of the SUMO-related operations
needed, such as specifying actions to be performed by vehicles and collecting
information on the network/vehicles for any given time step. In addition, the
base environment accepts states, actions, and rewards for the new step, and
outputs them to the reinforcement learning algorithm in ``rllab``, which in turn
trains the reinforcement learning agent(s) (i.e. the autonomous vehicles).

The inputs to the environment class are:

- **env\_params**: provides several environment and experiment-specific
  parameters. This includes specifying the parameters of the action space
  and relevant coefficients to the reward function.
- **sumo\_params**: used to pass the time step and sumo-specified safety
  modes, which constrain the dynamics of vehicles in the network to
  prevent crashes. In addition, this parameter may be used to specify whether to
  use sumo's gui during the experiment's runtime.
- **scenario**: The scenario class we created in `section 3`_

.. _section 3: creating-a-scenario_

By inheriting Flow's base environment, a custom environment can be created
by adding the following functions to the child class: ``action_space``,
``observation_space``, ``apply_rl_action``, ``get_state``, and
``compute_reward``, which are covered in the next few subsections.

4.2 Specifying an Action Space
------------------------------

The components of the action space are in the function conveniently
called ``action_space``; accordingly, we begin by defining this
function:

::

    @property
    def action_space(self):

The action space of an environment informs ``rllab`` on the number of
actions a given reinforcement learning agent can perform and the bounds on those
actions. In our single-lane ring road setting, autonomous vehicles can only
accelerate and decelerate, with each vehicle requiring a separate acceleration.
Moreover, their accelerations are bounded by maximum and minimum values
specified by the user.

Accordingly, we specify the number actions performed by the rl agent and bounds
of these actions as follows:

::

        num_acc_actions = self.vehicles.num_rl_vehicles
        acc_upper_bound = self.env_params.additional_params["max-acc"]
        acc_lower_bound = - abs(self.env_params.additional_params["max-deacc"])

Once the parameters of the action space are specified, the ``Box`` element
containing these attributes is defined as follows:

::

       acc_action_space = Box(low=acc_lower_bound, high=acc_upper_bound, shape=num_acc_actions)

       return acc_action_space

4.3 Specifying an Observation Space
-----------------------------------

The observation space of an environment represents the number and types
of observations that are provided to the reinforcement learning agent.
Assuming the system of vehicles are **fully** observable,
the observation space then consists of a vector of velocities :math:`v` and
absolute positions :math:`x` for each vehicle in the network.

We begin by defining our ``observation_space`` function:

::

    @property
    def observation_space(self):

In this function, we create two Box elements; one for the absolute
positions of the vehicles, and another for the speeds of the vehicles.
These values may range from zero to infinity, and there is a separate value
for each vehicles:

::

        speed = Box(low=0, high=np.inf, shape=(self.vehicles.num_vehicles,))
        absolute_pos = Box(low=0., high=np.inf, shape=(self.vehicles.num_vehicles,))

Finally, we combine the two ``Box`` elements using the Tuple method.
This tuple used at the output from the ``observation_space`` function:

::

        return Tuple([speed, absolute_pos])

4.4 Applying Actions to the Autonomous Vehicles
-----------------------------------------------

The function ``apply_rl_action`` acts as the bridge between ``rllab`` and
``sumo``, transforming commands specified by ``rllab`` in the action space into
actual action in the traffic scenario created within ``sumo``. This function
takes as an input the actions requested by ``rllab``, and sends the commands
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
the current time step, the variable ``self.vehicles`` can be used. This object
stores all sorts of information of the states of vehicles in the network, such
as their speed, edge, position, etc... This information can be accessed from
different "get" functions.

In order to create the necessary matrix of states, the function get\_state
loops through the vehicle ids of all vehicles in the network, and collects for
each vehicle its speed and absolute position:

::

        state = np.array([[self.vehicles.get_speed(veh_id),
                           self.vehicles.get_absolute_position(veh_id)]
                          for veh_id in self.ids])

        return state

.. _section 4.6:

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
as a mapping: :math:`r : S\times A \to R \geq 0`. This is done by subtracting
the deviation of the system from the desired velocity from the peak allowable
deviation from the desired velocity. Additionally, since the velocity of
vehicles are unbounded above, the reward is bounded below by zero, to ensure
nonnegativity.

Define :math:`v_{des}` as the desired velocity, :math:`1^k` a vector of ones of
length :math:`k`, :math:`n` as the number of vehicles in the system, and
:math:`v` as a vector of velocities. The reward function is formulated as:

.. math:: r(v) = \max{0, ||v_{des} \cdot 1^k ||_2 - || v - v_{des} \cdot 1^k ||_2}

**4.6.1 Using Built-in Reward Functions** Flow comes with several
built-in reward functions located in ``flow.core.rewards``.
In order to use these reward function, we begin by importing these reward
function at the top of the script:

::

    # Flow's built-in reward functions
    from flow.core import rewards

One reward function located in the ``rewards`` file is the function
``desired_velocity``, which computes the reward described in this
section. It takes as input the environment variable (``self``) and a
"fail" variables that specifies if the vehicles in the network
experiences any sort of crash, and is an element of the ``**kwargs``
variable. Returning to the ``compute_reward`` function, the reward may
be specified as follows:

::

        return rewards.desired_velocity(self, fail=kwargs["fail"])

**4.6.2 Building the Reward Function** In addition to using Flow's
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

        vel = self.vehicles.get_speed(self.ids)

        cost = vel - self.env_params.additional_params["target_velocity"]
        cost = np.linalg.norm(cost)

Finally, in order to ensure the value remains positive, we subtract this
deviation from the maximum allowable deviation, and clip the value from
below by zero.

::

        max_cost = np.array([self.env_params["target_velocity"]] * len(self.vehicles.num_vehicles))
        max_cost = np.linalg.norm(max_cost)

        return max(max_cost - cost, 0)

4.7 Registering the Environment as a Gym Environment
----------------------------------------------------

In order to run reinforcement learning experiments (see section 6), the
environment we created needs to be registered as a Gym Environment. In
order for Flow to register your environment as a Gym Environment, go
to ``flow/envs/__init__.py``, and add the following line:

::

    from <path to environment script>.my_environment import myEnvironment

5. Running an Experiment without Autonomy
=========================================

Once the classes described in sections 2, 3, and 4 are created, we are
now ready to run experiments with Flow. We begin by running an
experiment without any learning/autonomous agents. This experiment acts
as our control case, and helps us ensure that the system exhibits the
sorts of performance deficiencies we expect to witness. In the case of a
single-lane ring road, this deficiency is the phenomenon known as string
instability, in which vehicles begin producing stop-and-go waves among
themselves.

5.1 Importing the Necessary Modules
-----------------------------------

In order to run the experiment in the absence of autonomy, we will
create a ``SumoExperiment`` object. This variable takes as input the
environment and scenario classes developed in sections 3 and 4. Note
that the generator class is not needed by the experiment class, but
rather by the scenario class.

We begin by creating a new script in the same directory as that of the
generator and scenario classes titled ``my_control_experiment.py``. In
this script, we import the base experiment class, as well as the
generator, scenario, and environment subclasses we developed.

::

    # this is the base experiment class
    from flow.core.exp import SumoExperiment

    # these are the classes I created
    from ./my_generator import myGenerator
    from ./my_scenario import myScenario
    from ./my_environment import myEnvironment

    # for possible mathematical operation we may want to perform
    import numpy as np

In order to specify the inputs needed for each class, a few objects are also
imported from Flow.

::

    # input objects to my classes
    from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
    from flow.core.vehicles import Vehicles

Finally, in order to impose realistic vehicle dynamics on the vehicles in the
network, Flow possesses a few acceleration, lane-changing, and routing
controller classes. These classes are imported into the script as
follows:

::

    from flow.controllers.car_following_models import *
    from flow.controllers.lane_change_controllers import *
    from flow.controllers.routing_controllers import *

5.2 Setting Up the Environment and Scenario Classes
---------------------------------------------------

In order to initialize scenario and environment classes (as well as the
generator class which is initialized within the scenario), the inputs
for each class, must be must be specified. These inputs are:
``sumo_params``, ``vehicles``, ``env_params``, ``net_params``, and (optionally)
``initial_config``.

For the ``sumo_params`` input, we specify a time step of 0.1 s and turn on
sumo's gui to visualize the experiment as it happens:

::

    sumo_params = SumoParams(time_step=0.1, sumo_binary="sumo-gui")

Next, we initialize an empty vehicles object:

::

    vehicles = Vehicles()

22 human-driven vehicles are introduced to the vehicles object. These vehicles
are made to follow car-following dynamics defined by the Intelligent Driver
Model (IDM), and are rerouted every time they reach the end of their route
in order to ensure they stay in the ring indefinitely. This is done as follows:

::

    vehicles.add_vehicles(veh_id="idm",
                          acceleration_controller=(IDMController, {}),
                          routing_controller=(ContinuousRouter, {}),
                          num_vehicles=22)

For the ``env_params`` object, we specify the bounds of the action space.
We do this because ``rllab`` will continue to try to create an action space
object despite whether the outputted actions are used (such as in this base
experiment). These terms are added to the "additional_params" portion:

::

    additional_env_params = {"max-deacc": 3, "max-acc": 3}
    env_params = EnvParams(additional_params=additional_env_params)


In the  ``net_params`` object, we add the characteristic components of the
network. These values include: "radius", "lanes",
and "speed\_limit", and are added to the "additional_params" portion of the
network we descibed in `section 2`_.

.. _section 2: creating-a-generator_

::

    additional_net_params = {"length": 230, "lanes": 1, "speed_limit": 30}
    net_params = NetParams(additional_params=additional_net_params)


Note that, if `section 3.2`_ was not implemented when creating the scenario
class, an additional "length" component must be added to ``net_params``
as follows:

::

    net_params.additional_params["length"] = net_params.additional_params["radius"] * 2 * np.pi

Finally, in order to prevent the system from being perfectly symmetric, we add
a bunching component to the initial positioning of the vehicles, which is by
default "uniform":

::

    initial_config = InitialConfig(bunching=20)


Once all the necessary inputs are prepared, the scenario and environment
variables can be initialized. Moreover, naming the experiment
"ring\_road\_all\_human", the classes are created as followed:

::

    # create a scenario object
    scenario = myScenario("ring_road_all_human", myGenerator, vehicles, net_params,
                          initial_config)

    # create an environment object
    env = myEnvironment(env_params, sumo_params, scenario)

5.3 Setting up the Experiment Class
-----------------------------------

Once the environment and scenario classes are ready, the experiment
variable can be creating as follows:

::

    # creating an experiment object
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
``my_control_experiment.py`` from your IDE or from a terminal. After a
few seconds, a gui should appear on the screen with a circular road
network. Click on the play
button on the top-left corner of the gui, and the network will
be filled with vehicles, which then begin to accelerate.

As we can see, vehicles are not free-flowing in the ring. Instead, they seem to
generate stop-and-go waves in the ring, which forces all vehicles to slow down
constantly and prevents them from attaining their ideal equilibrium speeds.


6. Running an Experiment with Autonomy
======================================

Finally, we will attempt to add autonomous vehicles in the ring road. We
will begin by adding a single autonomous vehicles, in hopes that this
vehicle may be able to learn to attenuate the waves we witnessed in section 5.

6.1 Creating a Gym Environment
------------------------------

Unlike in section 5, we will not rely on Flow's ``SumoExperiment``
object to run experiments, but rather we will create a Gym Environment
and run it on ``rllab``.

Create a new script entitled
``my_rl_experiment.py`` and import the generator and scenario
subclasses, in addition to the dynamical model provided by Flow, as
you had done in section 5.1 for the control experiment:

::

    # these are the classes I created
    from ./my_generator import myGenerator
    from ./my_scenario import myScenario

    # for possible mathematical operation we may want to perform
    import numpy as np

    # acceleration and lane-changing controllers for human-driven vehicles
    from flow.controllers.car_following_models import *
    from flow.controllers.lane_change_controllers import *

A new controller that is used in this experiment and needed in the case
of mixed-autonomy is the ``RLController``, located in
``flow.controllers.rlcontroller``. Any types of vehicles with this
controller will act as reinforcement learning agent(s).

::

    from flow.controllers.rlcontroller import RLController

In additon, we will need several functions from ``rllab``:

::

    from rllab.envs.normalized_env import normalize
    from rllab.misc.instrument import run_experiment_lite
    from rllab.algos.trpo import TRPO
    from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
    from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
    from rllab.envs.gym_env import GymEnv

Next, we define a function called ``run_task`` that will
be used to create and run our gym environment:

::

    def run_task(v):

Similar to section 5, we must now define the necessary input variables
to the generator, scenario, and environment classes. These variable will
largely remain unchanged from section 5, but with the addition of a few
components.

For one, in ``sumo_params`` we will want to specify an aggressive
SUMO-defined speed mode for rl vehicles, which will prevent SUMO from enforcing
a safe velocity upper bound on the autonomous vehicle, but may lead to the
autonomous vehicles crashing into the vehicles ahead of them. This is
done by setting "rl\_sm" to "aggressive".

Moreover, in order to run rollouts with a max path length of 1500 steps
(i.e. 150 s), we set "num\_steps" in ``env_params`` to 1500. Also, in ordr to
satisfy the reward function we specified in `section 4.6`_, we set
"target\_velocity" in ``env_params`` to 8 m/s
(which far beyond the expected equilibrium velocity).

Finally we introduce an autonomous (rl) vehicle into the network by
reducing the number of human vehicles by 1 and add a element to the
``vehicles`` object to include a vehicle with the acceleration controller
``RLController``.

The final set of input variables are as follows:

::

        sumo_params = SumoParams(time_step=0.1, rl_speed_mode="aggressive",
                                 sumo_binary="sumo-gui")

        additional_env_params = {"target_velocity": 8, "max-deacc": 3, "max-acc": 3, "num_steps": 1000}
        env_params = EnvParams(additional_params=additional_env_params)

        additional_net_params = {"length": 230, "lanes": 1, "speed_limit": 30}
        net_params = NetParams(additional_params=additional_net_params)

        initial_config = InitialConfig(bunching=20)

        vehicles = Vehicles()
        vehicles.add_vehicles(veh_id="rl",
                              acceleration_controller=(RLController, {}),
                              routing_controller=(ContinuousRouter, {}),
                              num_vehicles=1)
        vehicles.add_vehicles(veh_id="human",
                              acceleration_controller=(IDMController, {}),
                              routing_controller=(ContinuousRouter, {}),
                              num_vehicles=21)

Creating the scenario does not change between this section and the last.
Calling our scenario "stabilizing-the-ring", the scenario class is
initialized as follows:

::

        scenario = myScenario("stabilizing-the-ring", myGenerator, vehicles, net_params,
                              initial_config)

The environment, however, is no longer defined in the same manner.
Instead, a variable called env\_name is specified with the name of the
environment you created, and the list of parameters are placed into a
tuple:

::

        env_name = "myEnvironment"
        pass_params = (env_name, sumo_params, type_params, env_params, net_params,
                       initial_config, scenario)

Then, the Gym Environment, parameterized by ``pass_params``, is initialized
as follows:

::

        env = GymEnv(env_name, record_video=False, register_params=pass_params)

6.2 Specifying the Necessary rllab Components
---------------------------------------------

We use linear feature baselines and Trust Region Policy Optimization for
learning the policy, with discount factor  :math:`\gamma = 0.999`, and step
size 0.01. A diagonal Gaussian MLP policy is used with hidden layers
(100, 50, 25) and tanh non-linearity. This is done within your script by adding
the following lines of code to the ``run_task`` function:

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
        seed=5,
        mode="local",
        exp_prefix="stabilizing-the-ring",
    )

Note that, when using Python editors such as PyCharm, it may be necessary to
specify the path to the location of ``rllab``'s python command within
``run_experiment_lite`` . This will look something similar to:

::

        python_command="<acaconda2_directory>/envs/rllab-distributed/bin/python3.5"

6.4 Running the Mixed-Autonomy Experiment
-----------------------------------------

We are finally ready to run our first experiment with reinforcement learning
autonomous agents! Run the script and click on the "Play" button on sumo's gui
as you had done in section 5. The experiment will now run for a maximum of 300
iterations (as we had specified); however, the experiments converges much sooner.
In fact, by around the 150th iteration, we notice that the vehicle had learned
to stop crashing completely, and that the vehicles in the ring seem to be
completely free-flowing, without the nuisance of stop-and-go waves.
