flow.envs
=========

.. raw:: html

    <div class="section" id="flow-envs-package">
    <div class="section" id="submodules">
    <h2>Submodules<a class="headerlink" href="#submodules" title="Permalink to this headline">¶</a></h2>
    </div>
    <div class="section" id="module-flow.envs.base_env">
    <span id="flow-envs-base-env-module"></span><h2>flow.envs.base_env module<a class="headerlink" href="#module-flow.envs.base_env" title="Permalink to this headline">¶</a></h2>
    <dl class="class">
    <dt id="flow.envs.base_env.SumoEnvironment">
    <em class="property">class </em><code class="descclassname">flow.envs.base_env.</code><code class="descname">SumoEnvironment</code><span class="sig-paren">(</span><em>env_params</em>, <em>sumo_params</em>, <em>scenario</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/base_env.html#SumoEnvironment"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.base_env.SumoEnvironment" title="Permalink to this definition">¶</a></dt>
    <dd><p>Bases: <code class="xref py py-class docutils literal"><span class="pre">gym.core.Env</span></code>, <code class="xref py py-class docutils literal"><span class="pre">rllab.core.serializable.Serializable</span></code></p>
    <dl class="attribute">
    <dt id="flow.envs.base_env.SumoEnvironment.action_space">
    <code class="descname">action_space</code><a class="headerlink" href="#flow.envs.base_env.SumoEnvironment.action_space" title="Permalink to this definition">¶</a></dt>
    <dd><p>Identifies the dimensions and bounds of the action space (needed for
    rllab environments).
    MUST BE implemented in new environments.</p>
    <table class="docutils field-list" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
    <tr class="field-odd field"><th class="field-name">Yields:</th><td class="field-body"><em>rllab Box or Tuple type</em> – a bounded box depicting the shape and bounds of the action space</td>
    </tr>
    </tbody>
    </table>
    </dd></dl>

    <dl class="method">
    <dt id="flow.envs.base_env.SumoEnvironment.additional_command">
    <code class="descname">additional_command</code><span class="sig-paren">(</span><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/base_env.html#SumoEnvironment.additional_command"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.base_env.SumoEnvironment.additional_command" title="Permalink to this definition">¶</a></dt>
    <dd><p>Additional commands that may be performed before a simulation step.</p>
    </dd></dl>

    <dl class="method">
    <dt id="flow.envs.base_env.SumoEnvironment.apply_acceleration">
    <code class="descname">apply_acceleration</code><span class="sig-paren">(</span><em>veh_ids</em>, <em>acc</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/base_env.html#SumoEnvironment.apply_acceleration"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.base_env.SumoEnvironment.apply_acceleration" title="Permalink to this definition">¶</a></dt>
    <dd><p>Applies the acceleration requested by a vehicle in sumo. Note that, if
    the sumo-specified speed mode of the vehicle is not “aggressive”, the
    acceleration may be clipped by some saftey velocity or maximum possible
    acceleration.</p>
    <table class="docutils field-list" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
    <tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><ul class="first last simple">
    <li><strong>veh_ids</strong> (<em>list of strings</em>) – vehicles IDs associated with the requested accelerations</li>
    <li><strong>acc</strong> (<em>numpy array or list of float</em>) – requested accelerations from the vehicles</li>
    </ul>
    </td>
    </tr>
    </tbody>
    </table>
    </dd></dl>

    <dl class="method">
    <dt id="flow.envs.base_env.SumoEnvironment.apply_lane_change">
    <code class="descname">apply_lane_change</code><span class="sig-paren">(</span><em>veh_ids</em>, <em>direction=None</em>, <em>target_lane=None</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/base_env.html#SumoEnvironment.apply_lane_change"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.base_env.SumoEnvironment.apply_lane_change" title="Permalink to this definition">¶</a></dt>
    <dd><p>Applies an instantaneous lane-change to a set of vehicles, while
    preventing vehicles from moving to lanes that do not exist.</p>
    <table class="docutils field-list" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
    <tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><ul class="first">
    <li><p class="first"><strong>veh_ids</strong> (<em>list of strings</em>) – vehicles IDs associated with the requested accelerations</p>
    </li>
    <li><p class="first"><strong>direction</strong> (<em>list of int (-1, 0, or 1), optional</em>) –</p>
    <dl class="docutils">
    <dt>-1: lane change to the right</dt>
    <dd><p class="first last">0: no lange change
    1: lane change to the left</p>
    </dd>
    </dl>
    </li>
    <li><p class="first"><strong>target_lane</strong> (<em>list of int, optional</em>) – lane indices the vehicles should lane-change to in the next step</p>
    </li>
    </ul>
    </td>
    </tr>
    <tr class="field-even field"><th class="field-name">Raises:</th><td class="field-body"><p class="first last"><code class="xref py py-exc docutils literal"><span class="pre">ValueError</span></code> – If either both or none of “direction” and “target_lane” are provided
    as inputs. Only one should be provided at a time.</p>
    </td>
    </tr>
    </tbody>
    </table>
    </dd></dl>

    <dl class="method">
    <dt id="flow.envs.base_env.SumoEnvironment.apply_rl_actions">
    <code class="descname">apply_rl_actions</code><span class="sig-paren">(</span><em>rl_actions</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/base_env.html#SumoEnvironment.apply_rl_actions"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.base_env.SumoEnvironment.apply_rl_actions" title="Permalink to this definition">¶</a></dt>
    <dd><p>Specifies the actions to be performed by rl_vehicles</p>
    <table class="docutils field-list" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
    <tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><strong>rl_actions</strong> (<em>numpy ndarray</em>) – list of actions provided by the RL algorithm</td>
    </tr>
    </tbody>
    </table>
    </dd></dl>

    <dl class="method">
    <dt id="flow.envs.base_env.SumoEnvironment.choose_routes">
    <code class="descname">choose_routes</code><span class="sig-paren">(</span><em>veh_ids</em>, <em>route_choices</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/base_env.html#SumoEnvironment.choose_routes"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.base_env.SumoEnvironment.choose_routes" title="Permalink to this definition">¶</a></dt>
    <dd><p>Updates the route choice of vehicles in the network.</p>
    <table class="docutils field-list" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
    <tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><ul class="first last simple">
    <li><strong>veh_ids</strong> (<em>list</em>) – list of vehicle identifiers</li>
    <li><strong>route_choices</strong> (<em>numpy array or list of floats</em>) – list of edges the vehicle wishes to traverse, starting with the edge
    the vehicle is currently on. If a value of None is provided, the
    vehicle does not update its route</li>
    </ul>
    </td>
    </tr>
    </tbody>
    </table>
    </dd></dl>

    <dl class="method">
    <dt id="flow.envs.base_env.SumoEnvironment.compute_reward">
    <code class="descname">compute_reward</code><span class="sig-paren">(</span><em>state</em>, <em>rl_actions</em>, <em>**kwargs</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/base_env.html#SumoEnvironment.compute_reward"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.base_env.SumoEnvironment.compute_reward" title="Permalink to this definition">¶</a></dt>
    <dd><p>Reward function for RL.
    MUST BE implemented in new environments.</p>
    <table class="docutils field-list" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
    <tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><ul class="first simple">
    <li><strong>state</strong> (<em>numpy ndarray</em>) – state of all the vehicles in the simulation</li>
    <li><strong>rl_actions</strong> (<em>numpy ndarray</em>) – actions performed by rl vehicles</li>
    <li><strong>kwargs</strong> (<em>dictionary</em>) – other parameters of interest. Contains a “fail” element, which
    is True if a vehicle crashed, and False otherwise</li>
    </ul>
    </td>
    </tr>
    <tr class="field-even field"><th class="field-name">Returns:</th><td class="field-body"><p class="first"><strong>reward</strong></p>
    </td>
    </tr>
    <tr class="field-odd field"><th class="field-name">Return type:</th><td class="field-body"><p class="first last">float</p>
    </td>
    </tr>
    </tbody>
    </table>
    </dd></dl>

    <dl class="method">
    <dt id="flow.envs.base_env.SumoEnvironment.get_headway_dict">
    <code class="descname">get_headway_dict</code><span class="sig-paren">(</span><em>network_observations</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/base_env.html#SumoEnvironment.get_headway_dict"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.base_env.SumoEnvironment.get_headway_dict" title="Permalink to this definition">¶</a></dt>
    <dd><p>Collects the headways, leaders, and followers of all vehicles at once.
    The base environment does by using traci calls.</p>
    <table class="docutils field-list" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
    <tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><strong>network_observations</strong> (<em>dictionary</em>) – key = vehicle IDs
    elements = variable state properties of the vehicle (including
    headway)</td>
    </tr>
    <tr class="field-even field"><th class="field-name">Yields:</th><td class="field-body"><em>dictionary</em> – key = vehicle IDs
    elements = headway, leader id, and follower id for the vehicle</td>
    </tr>
    </tbody>
    </table>
    </dd></dl>

    <dl class="method">
    <dt id="flow.envs.base_env.SumoEnvironment.get_state">
    <code class="descname">get_state</code><span class="sig-paren">(</span><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/base_env.html#SumoEnvironment.get_state"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.base_env.SumoEnvironment.get_state" title="Permalink to this definition">¶</a></dt>
    <dd><p>Returns the state of the simulation as perceived by the learning agent.
    MUST BE implemented in new environments.</p>
    <table class="docutils field-list" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
    <tr class="field-odd field"><th class="field-name">Returns:</th><td class="field-body"><strong>state</strong> – information on the state of the vehicles, which is provided to the
    agent</td>
    </tr>
    <tr class="field-even field"><th class="field-name">Return type:</th><td class="field-body">numpy ndarray</td>
    </tr>
    </tbody>
    </table>
    </dd></dl>

    <dl class="method">
    <dt id="flow.envs.base_env.SumoEnvironment.get_x_by_id">
    <code class="descname">get_x_by_id</code><span class="sig-paren">(</span><em>veh_id</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/base_env.html#SumoEnvironment.get_x_by_id"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.base_env.SumoEnvironment.get_x_by_id" title="Permalink to this definition">¶</a></dt>
    <dd><p>Provides a 1-dimensional representation of the position of a vehicle
    in the network.</p>
    <table class="docutils field-list" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
    <tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><strong>veh_id</strong> (<em>string</em>) – vehicle identifier</td>
    </tr>
    <tr class="field-even field"><th class="field-name">Yields:</th><td class="field-body"><em>float</em> – position of a vehicle relative to a certain reference.</td>
    </tr>
    </tbody>
    </table>
    </dd></dl>

    <dl class="attribute">
    <dt id="flow.envs.base_env.SumoEnvironment.observation_space">
    <code class="descname">observation_space</code><a class="headerlink" href="#flow.envs.base_env.SumoEnvironment.observation_space" title="Permalink to this definition">¶</a></dt>
    <dd><p>Identifies the dimensions and bounds of the observation space (needed
    for rllab environments).
    MUST BE implemented in new environments.</p>
    <table class="docutils field-list" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
    <tr class="field-odd field"><th class="field-name">Yields:</th><td class="field-body"><em>rllab Box or Tuple type</em> – a bounded box depicting the shape and bounds of the observation
    space</td>
    </tr>
    </tbody>
    </table>
    </dd></dl>

    <dl class="method">
    <dt id="flow.envs.base_env.SumoEnvironment.restart_sumo">
    <code class="descname">restart_sumo</code><span class="sig-paren">(</span><em>sumo_params</em>, <em>sumo_binary=None</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/base_env.html#SumoEnvironment.restart_sumo"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.base_env.SumoEnvironment.restart_sumo" title="Permalink to this definition">¶</a></dt>
    <dd><p>Restarts an already initialized environment. Used when visualizing a
    rollout.</p>
    </dd></dl>

    <dl class="method">
    <dt id="flow.envs.base_env.SumoEnvironment.set_lane_change_mode">
    <code class="descname">set_lane_change_mode</code><span class="sig-paren">(</span><em>veh_id</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/base_env.html#SumoEnvironment.set_lane_change_mode"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.base_env.SumoEnvironment.set_lane_change_mode" title="Permalink to this definition">¶</a></dt>
    <dd><p>Sets the SUMO-defined lane-change mode with Traci. This is used to
    constrain lane-changing actions.</p>
    <dl class="docutils">
    <dt>The available lane-changing modes are as follows:</dt>
    <dd><ul class="first last simple">
    <li>default: Human and RL cars can only safely change into lanes</li>
    <li>“strategic”: Human cars make lane changes in accordance with SUMO to
    provide speed boosts</li>
    <li>“no_lat_collide”: RL cars can lane change into any space, no matter
    how likely it is to crash</li>
    <li>“aggressive”: RL cars are not limited by sumo with regard to their
    lane-change actions, and can crash longitudinally</li>
    </ul>
    </dd>
    </dl>
    <table class="docutils field-list" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
    <tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><strong>veh_id</strong> (<em>string</em>) – vehicle identifier</td>
    </tr>
    </tbody>
    </table>
    </dd></dl>

    <dl class="method">
    <dt id="flow.envs.base_env.SumoEnvironment.set_speed_mode">
    <code class="descname">set_speed_mode</code><span class="sig-paren">(</span><em>veh_id</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/base_env.html#SumoEnvironment.set_speed_mode"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.base_env.SumoEnvironment.set_speed_mode" title="Permalink to this definition">¶</a></dt>
    <dd><p>Sets the SUMO-defined speed mode with Traci. This is used to constrain
    acceleration actions.</p>
    <dl class="docutils">
    <dt>The available speed modes are as follows:</dt>
    <dd><ul class="first last simple">
    <li>“no_collide” (default): Human and RL cars are preventing from
    reaching speeds that may cause crashes (also serves as a failsafe).</li>
    <li>“aggressive”: Human and RL cars are not limited by sumo with regard
    to their accelerations, and can crash longitudinally</li>
    </ul>
    </dd>
    </dl>
    <table class="docutils field-list" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
    <tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><strong>veh_id</strong> (<em>string</em>) – vehicle identifier</td>
    </tr>
    </tbody>
    </table>
    </dd></dl>

    <dl class="method">
    <dt id="flow.envs.base_env.SumoEnvironment.setup_initial_state">
    <code class="descname">setup_initial_state</code><span class="sig-paren">(</span><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/base_env.html#SumoEnvironment.setup_initial_state"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.base_env.SumoEnvironment.setup_initial_state" title="Permalink to this definition">¶</a></dt>
    <dd><p>Returns information on the initial state of the vehicles in the network,
    to be used upon reset.
    Also adds initial state information to the self.vehicles class and
    starts a subscription with sumo to collect state information each step.</p>
    <table class="docutils field-list" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
    <tr class="field-odd field"><th class="field-name">Returns:</th><td class="field-body"><ul class="simple">
    <li><strong>initial_observations</strong> (<em>dictionary</em>) – key = vehicles IDs
    value = state describing car at the start of the rollout</li>
    <li><strong>initial_state</strong> (<em>dictionary</em>) – key = vehicles IDs
    value = sparse state information (only what is needed to add a
    vehicle in a sumo network with traci)</li>
    </ul>
    </td>
    </tr>
    </tbody>
    </table>
    </dd></dl>

    <dl class="method">
    <dt id="flow.envs.base_env.SumoEnvironment.sort_by_position">
    <code class="descname">sort_by_position</code><span class="sig-paren">(</span><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/base_env.html#SumoEnvironment.sort_by_position"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.base_env.SumoEnvironment.sort_by_position" title="Permalink to this definition">¶</a></dt>
    <dd><p>Sorts the vehicle ids of vehicles in the network by position.
    The base environment does this by sorting vehicles by their absolute
    position, as specified by the “get_x_by_id” function.</p>
    <table class="docutils field-list" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
    <tr class="field-odd field"><th class="field-name">Returns:</th><td class="field-body"><ul class="simple">
    <li><strong>sorted_ids</strong> (<em>list</em>) – a list of all vehicle IDs sorted by position</li>
    <li><strong>sorted_extra_data</strong> (<em>list or tuple</em>) – an extra component (list, tuple, etc…) containing extra sorted
    data, such as positions. If no extra component is needed, a value
    of None should be returned</li>
    </ul>
    </td>
    </tr>
    </tbody>
    </table>
    </dd></dl>

    <dl class="method">
    <dt id="flow.envs.base_env.SumoEnvironment.start_sumo">
    <code class="descname">start_sumo</code><span class="sig-paren">(</span><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/base_env.html#SumoEnvironment.start_sumo"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.base_env.SumoEnvironment.start_sumo" title="Permalink to this definition">¶</a></dt>
    <dd><p>Starts a sumo instance using the configuration files created by the
    generator class. Also initializes a traci connection to interface with
    sumo from Python.</p>
    </dd></dl>

    <dl class="method">
    <dt id="flow.envs.base_env.SumoEnvironment.terminate">
    <code class="descname">terminate</code><span class="sig-paren">(</span><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/base_env.html#SumoEnvironment.terminate"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.base_env.SumoEnvironment.terminate" title="Permalink to this definition">¶</a></dt>
    <dd><p>Closes the TraCI I/O connection. Should be done at end of every
    experiment. Must be in Environment because the environment opens the
    TraCI connection.</p>
    </dd></dl>

    </dd></dl>

    </div>
    <div class="section" id="module-flow.envs.intersection_env">
    <span id="flow-envs-intersection-env-module"></span><h2>flow.envs.intersection_env module<a class="headerlink" href="#module-flow.envs.intersection_env" title="Permalink to this headline">¶</a></h2>
    <dl class="class">
    <dt id="flow.envs.intersection_env.IntersectionEnvironment">
    <em class="property">class </em><code class="descclassname">flow.envs.intersection_env.</code><code class="descname">IntersectionEnvironment</code><span class="sig-paren">(</span><em>env_params</em>, <em>sumo_params</em>, <em>scenario</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/intersection_env.html#IntersectionEnvironment"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.intersection_env.IntersectionEnvironment" title="Permalink to this definition">¶</a></dt>
    <dd><p>Bases: <a class="reference internal" href="#flow.envs.base_env.SumoEnvironment" title="flow.envs.base_env.SumoEnvironment"><code class="xref py py-class docutils literal"><span class="pre">flow.envs.base_env.SumoEnvironment</span></code></a></p>
    <p>A class that may be used to design environments with intersections. Allows
    the user to calculate the distance a vehicle is from the nearest
    intersection, assuming the function “specify_intersection_edge_starts” in
    the scenario class is properly prepared.</p>
    <dl class="method">
    <dt id="flow.envs.intersection_env.IntersectionEnvironment.find_intersection_dist">
    <code class="descname">find_intersection_dist</code><span class="sig-paren">(</span><em>veh_id</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/intersection_env.html#IntersectionEnvironment.find_intersection_dist"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.intersection_env.IntersectionEnvironment.find_intersection_dist" title="Permalink to this definition">¶</a></dt>
    <dd></dd></dl>

    <dl class="method">
    <dt id="flow.envs.intersection_env.IntersectionEnvironment.get_distance_to_intersection">
    <code class="descname">get_distance_to_intersection</code><span class="sig-paren">(</span><em>veh_ids</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/intersection_env.html#IntersectionEnvironment.get_distance_to_intersection"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.intersection_env.IntersectionEnvironment.get_distance_to_intersection" title="Permalink to this definition">¶</a></dt>
    <dd><p>Determines the smallest distance from the current vehicle’s position to
    any of the intersections.</p>
    <table class="docutils field-list" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
    <tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><strong>veh_ids</strong> (<em>str</em>) – vehicle identifier</td>
    </tr>
    <tr class="field-even field"><th class="field-name">Yields:</th><td class="field-body"><em>tup</em> – 1st element: distance to closest intersection
    2nd element: intersection ID (which also specifies which side of the
    intersection the vehicle will be arriving at)</td>
    </tr>
    </tbody>
    </table>
    </dd></dl>

    <dl class="method">
    <dt id="flow.envs.intersection_env.IntersectionEnvironment.sort_by_intersection_dist">
    <code class="descname">sort_by_intersection_dist</code><span class="sig-paren">(</span><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/intersection_env.html#IntersectionEnvironment.sort_by_intersection_dist"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.intersection_env.IntersectionEnvironment.sort_by_intersection_dist" title="Permalink to this definition">¶</a></dt>
    <dd><p>Sorts the vehicle ids of vehicles in the network by their distance to
    the intersection.
    The base environment does this by sorting vehicles by their distance to
    intersection, as specified by the “get_distance_to_intersection”
    function.</p>
    <table class="docutils field-list" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
    <tr class="field-odd field"><th class="field-name">Returns:</th><td class="field-body"><ul class="simple">
    <li><strong>sorted_ids</strong> (<em>list</em>) – a list of all vehicle IDs sorted by position</li>
    <li><strong>sorted_extra_data</strong> (<em>list or tuple</em>) – an extra component (list, tuple, etc…) containing extra sorted
    data, such as positions. If no extra component is needed, a value
    of None should be returned</li>
    </ul>
    </td>
    </tr>
    </tbody>
    </table>
    </dd></dl>

    </dd></dl>

    </div>
    <div class="section" id="module-flow.envs.lane_changing">
    <span id="flow-envs-lane-changing-module"></span><h2>flow.envs.lane_changing module<a class="headerlink" href="#module-flow.envs.lane_changing" title="Permalink to this headline">¶</a></h2>
    <dl class="class">
    <dt id="flow.envs.lane_changing.LaneChangeOnlyEnvironment">
    <em class="property">class </em><code class="descclassname">flow.envs.lane_changing.</code><code class="descname">LaneChangeOnlyEnvironment</code><span class="sig-paren">(</span><em>env_params</em>, <em>sumo_params</em>, <em>scenario</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/lane_changing.html#LaneChangeOnlyEnvironment"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.lane_changing.LaneChangeOnlyEnvironment" title="Permalink to this definition">¶</a></dt>
    <dd><p>Bases: <a class="reference internal" href="#flow.envs.lane_changing.SimpleLaneChangingAccelerationEnvironment" title="flow.envs.lane_changing.SimpleLaneChangingAccelerationEnvironment"><code class="xref py py-class docutils literal"><span class="pre">flow.envs.lane_changing.SimpleLaneChangingAccelerationEnvironment</span></code></a></p>
    <p>Am extension of SimpleLaneChangingAccelerationEnvironment. Autonomous
    vehicles in this environment can only make lane-changing decisions. Their
    accelerations, on the other hand, are controlled by an human car-following
    model specified under “rl_acc_controller” in the in additional_params
    attribute of env_params.</p>
    <dl class="attribute">
    <dt id="flow.envs.lane_changing.LaneChangeOnlyEnvironment.action_space">
    <code class="descname">action_space</code><a class="headerlink" href="#flow.envs.lane_changing.LaneChangeOnlyEnvironment.action_space" title="Permalink to this definition">¶</a></dt>
    <dd><p>See parent class</p>
    <p>Actions are: a continuous direction for each rl vehicle</p>
    </dd></dl>

    <dl class="method">
    <dt id="flow.envs.lane_changing.LaneChangeOnlyEnvironment.apply_rl_actions">
    <code class="descname">apply_rl_actions</code><span class="sig-paren">(</span><em>actions</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/lane_changing.html#LaneChangeOnlyEnvironment.apply_rl_actions"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.lane_changing.LaneChangeOnlyEnvironment.apply_rl_actions" title="Permalink to this definition">¶</a></dt>
    <dd><p>see parent class</p>
    <p>Actions are applied to rl vehicles as follows:
    - accelerations are derived using the user-specified accel controller
    - lane-change commands are collected from rllab</p>
    </dd></dl>

    </dd></dl>

    <dl class="class">
    <dt id="flow.envs.lane_changing.SimpleLaneChangingAccelerationEnvironment">
    <em class="property">class </em><code class="descclassname">flow.envs.lane_changing.</code><code class="descname">SimpleLaneChangingAccelerationEnvironment</code><span class="sig-paren">(</span><em>env_params</em>, <em>sumo_params</em>, <em>scenario</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/lane_changing.html#SimpleLaneChangingAccelerationEnvironment"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.lane_changing.SimpleLaneChangingAccelerationEnvironment" title="Permalink to this definition">¶</a></dt>
    <dd><p>Bases: <a class="reference internal" href="#flow.envs.base_env.SumoEnvironment" title="flow.envs.base_env.SumoEnvironment"><code class="xref py py-class docutils literal"><span class="pre">flow.envs.base_env.SumoEnvironment</span></code></a></p>
    <p>Fully functional environment for multi lane closed loop settings. Takes in
    an <em>acceleration</em> and <em>lane-change</em> as an action. Reward function is
    negative norm of the difference between the velocities of each vehicle, and
    the target velocity. State function is a vector of the velocities and
    absolute positions for each vehicle.</p>
    <dl class="attribute">
    <dt id="flow.envs.lane_changing.SimpleLaneChangingAccelerationEnvironment.action_space">
    <code class="descname">action_space</code><a class="headerlink" href="#flow.envs.lane_changing.SimpleLaneChangingAccelerationEnvironment.action_space" title="Permalink to this definition">¶</a></dt>
    <dd><p>See parent class</p>
    <dl class="docutils">
    <dt>Actions are:</dt>
    <dd><ul class="first last simple">
    <li>a (continuous) acceleration from max-deacc to max-acc</li>
    <li>a (continuous) lane-change action from -1 to 1, used to determine the
    lateral direction the vehicle will take.</li>
    </ul>
    </dd>
    </dl>
    </dd></dl>

    <dl class="method">
    <dt id="flow.envs.lane_changing.SimpleLaneChangingAccelerationEnvironment.apply_rl_actions">
    <code class="descname">apply_rl_actions</code><span class="sig-paren">(</span><em>actions</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/lane_changing.html#SimpleLaneChangingAccelerationEnvironment.apply_rl_actions"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.lane_changing.SimpleLaneChangingAccelerationEnvironment.apply_rl_actions" title="Permalink to this definition">¶</a></dt>
    <dd><p>See parent class</p>
    <p>Takes a tuple and applies a lane change or acceleration. if a lane
    change is applied, don’t issue any commands for the duration of the lane
    change and return negative rewards for actions during that lane change.
    if a lane change isn’t applied, and sufficient time has passed, issue an
    acceleration like normal.</p>
    </dd></dl>

    <dl class="method">
    <dt id="flow.envs.lane_changing.SimpleLaneChangingAccelerationEnvironment.compute_reward">
    <code class="descname">compute_reward</code><span class="sig-paren">(</span><em>state</em>, <em>rl_actions</em>, <em>**kwargs</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/lane_changing.html#SimpleLaneChangingAccelerationEnvironment.compute_reward"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.lane_changing.SimpleLaneChangingAccelerationEnvironment.compute_reward" title="Permalink to this definition">¶</a></dt>
    <dd><p>See parent class</p>
    <p>The reward function is negative norm of the difference between the
    velocities of each vehicle, and the target velocity. Also, a small
    penalty is added for rl lane changes in order to encourage mimizing
    lane-changing action.</p>
    </dd></dl>

    <dl class="method">
    <dt id="flow.envs.lane_changing.SimpleLaneChangingAccelerationEnvironment.get_state">
    <code class="descname">get_state</code><span class="sig-paren">(</span><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/lane_changing.html#SimpleLaneChangingAccelerationEnvironment.get_state"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.lane_changing.SimpleLaneChangingAccelerationEnvironment.get_state" title="Permalink to this definition">¶</a></dt>
    <dd><p>See parent class</p>
    <p>The state is an array the velocities, absolute positions, and lane
    numbers for each vehicle.</p>
    </dd></dl>

    <dl class="attribute">
    <dt id="flow.envs.lane_changing.SimpleLaneChangingAccelerationEnvironment.observation_space">
    <code class="descname">observation_space</code><a class="headerlink" href="#flow.envs.lane_changing.SimpleLaneChangingAccelerationEnvironment.observation_space" title="Permalink to this definition">¶</a></dt>
    <dd><p>See parent class</p>
    <p>An observation consists of the velocity, absolute position, and lane
    index of each vehicle in the fleet</p>
    </dd></dl>

    </dd></dl>

    </div>
    <div class="section" id="module-flow.envs.loop_accel">
    <span id="flow-envs-loop-accel-module"></span><h2>flow.envs.loop_accel module<a class="headerlink" href="#module-flow.envs.loop_accel" title="Permalink to this headline">¶</a></h2>
    <dl class="class">
    <dt id="flow.envs.loop_accel.SimpleAccelerationEnvironment">
    <em class="property">class </em><code class="descclassname">flow.envs.loop_accel.</code><code class="descname">SimpleAccelerationEnvironment</code><span class="sig-paren">(</span><em>env_params</em>, <em>sumo_params</em>, <em>scenario</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/loop_accel.html#SimpleAccelerationEnvironment"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.loop_accel.SimpleAccelerationEnvironment" title="Permalink to this definition">¶</a></dt>
    <dd><p>Bases: <a class="reference internal" href="#flow.envs.base_env.SumoEnvironment" title="flow.envs.base_env.SumoEnvironment"><code class="xref py py-class docutils literal"><span class="pre">flow.envs.base_env.SumoEnvironment</span></code></a></p>
    <p>Fully functional environment for single lane closed loop settings. Takes in
    an <em>acceleration</em> as an action. Reward function is negative norm of the
    difference between the velocities of each vehicle, and the target velocity.
    State function is a vector of the velocities and absolute positions for each
    vehicle.</p>
    <dl class="attribute">
    <dt id="flow.envs.loop_accel.SimpleAccelerationEnvironment.action_space">
    <code class="descname">action_space</code><a class="headerlink" href="#flow.envs.loop_accel.SimpleAccelerationEnvironment.action_space" title="Permalink to this definition">¶</a></dt>
    <dd><p>See parent class</p>
    <p>Actions are a set of accelerations from max-deacc to max-acc for each
    rl vehicle.</p>
    </dd></dl>

    <dl class="method">
    <dt id="flow.envs.loop_accel.SimpleAccelerationEnvironment.apply_rl_actions">
    <code class="descname">apply_rl_actions</code><span class="sig-paren">(</span><em>rl_actions</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/loop_accel.html#SimpleAccelerationEnvironment.apply_rl_actions"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.loop_accel.SimpleAccelerationEnvironment.apply_rl_actions" title="Permalink to this definition">¶</a></dt>
    <dd><p>See parent class</p>
    <p>Accelerations are applied to rl vehicles in accordance with the commands
    provided by rllab. These actions may be altered by flow’s failsafes or
    sumo-defined speed modes.</p>
    </dd></dl>

    <dl class="method">
    <dt id="flow.envs.loop_accel.SimpleAccelerationEnvironment.compute_reward">
    <code class="descname">compute_reward</code><span class="sig-paren">(</span><em>state</em>, <em>rl_actions</em>, <em>**kwargs</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/loop_accel.html#SimpleAccelerationEnvironment.compute_reward"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.loop_accel.SimpleAccelerationEnvironment.compute_reward" title="Permalink to this definition">¶</a></dt>
    <dd><p>See parent class</p>
    </dd></dl>

    <dl class="method">
    <dt id="flow.envs.loop_accel.SimpleAccelerationEnvironment.get_state">
    <code class="descname">get_state</code><span class="sig-paren">(</span><em>**kwargs</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/loop_accel.html#SimpleAccelerationEnvironment.get_state"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.loop_accel.SimpleAccelerationEnvironment.get_state" title="Permalink to this definition">¶</a></dt>
    <dd><p>See parent class</p>
    <p>The state is an array of velocities and absolute positions for each
    vehicle</p>
    </dd></dl>

    <dl class="attribute">
    <dt id="flow.envs.loop_accel.SimpleAccelerationEnvironment.observation_space">
    <code class="descname">observation_space</code><a class="headerlink" href="#flow.envs.loop_accel.SimpleAccelerationEnvironment.observation_space" title="Permalink to this definition">¶</a></dt>
    <dd><p>See parent class</p>
    <p>An observation is an array the velocities and absolute positions for
    each vehicle</p>
    </dd></dl>

    </dd></dl>

    <dl class="class">
    <dt id="flow.envs.loop_accel.SimpleMultiAgentAccelerationEnvironment">
    <em class="property">class </em><code class="descclassname">flow.envs.loop_accel.</code><code class="descname">SimpleMultiAgentAccelerationEnvironment</code><span class="sig-paren">(</span><em>env_params</em>, <em>sumo_params</em>, <em>scenario</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/loop_accel.html#SimpleMultiAgentAccelerationEnvironment"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.loop_accel.SimpleMultiAgentAccelerationEnvironment" title="Permalink to this definition">¶</a></dt>
    <dd><p>Bases: <a class="reference internal" href="#flow.envs.loop_accel.SimpleAccelerationEnvironment" title="flow.envs.loop_accel.SimpleAccelerationEnvironment"><code class="xref py py-class docutils literal"><span class="pre">flow.envs.loop_accel.SimpleAccelerationEnvironment</span></code></a></p>
    <p>An extension of SimpleAccelerationEnvironment which treats each autonomous
    vehicles as a separate rl agent, thereby allowing autonomous vehicles to be
    trained in multi-agent settings.</p>
    <dl class="attribute">
    <dt id="flow.envs.loop_accel.SimpleMultiAgentAccelerationEnvironment.action_space">
    <code class="descname">action_space</code><a class="headerlink" href="#flow.envs.loop_accel.SimpleMultiAgentAccelerationEnvironment.action_space" title="Permalink to this definition">¶</a></dt>
    <dd><p>See parent class</p>
    <p>Actions are a set of accelerations from max-deacc to max-acc for each
    rl vehicle.</p>
    </dd></dl>

    <dl class="method">
    <dt id="flow.envs.loop_accel.SimpleMultiAgentAccelerationEnvironment.compute_reward">
    <code class="descname">compute_reward</code><span class="sig-paren">(</span><em>state</em>, <em>rl_actions</em>, <em>**kwargs</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/loop_accel.html#SimpleMultiAgentAccelerationEnvironment.compute_reward"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.loop_accel.SimpleMultiAgentAccelerationEnvironment.compute_reward" title="Permalink to this definition">¶</a></dt>
    <dd><p>See parent class</p>
    </dd></dl>

    <dl class="method">
    <dt id="flow.envs.loop_accel.SimpleMultiAgentAccelerationEnvironment.get_state">
    <code class="descname">get_state</code><span class="sig-paren">(</span><em>**kwargs</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/loop_accel.html#SimpleMultiAgentAccelerationEnvironment.get_state"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.loop_accel.SimpleMultiAgentAccelerationEnvironment.get_state" title="Permalink to this definition">¶</a></dt>
    <dd><p>See parent class
    The state is an array the velocities and absolute positions for
    each vehicle.</p>
    </dd></dl>

    <dl class="attribute">
    <dt id="flow.envs.loop_accel.SimpleMultiAgentAccelerationEnvironment.observation_space">
    <code class="descname">observation_space</code><a class="headerlink" href="#flow.envs.loop_accel.SimpleMultiAgentAccelerationEnvironment.observation_space" title="Permalink to this definition">¶</a></dt>
    <dd><p>See parent class</p>
    </dd></dl>

    </dd></dl>

    <dl class="class">
    <dt id="flow.envs.loop_accel.SimplePartiallyObservableEnvironment">
    <em class="property">class </em><code class="descclassname">flow.envs.loop_accel.</code><code class="descname">SimplePartiallyObservableEnvironment</code><span class="sig-paren">(</span><em>env_params</em>, <em>sumo_params</em>, <em>scenario</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/loop_accel.html#SimplePartiallyObservableEnvironment"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.loop_accel.SimplePartiallyObservableEnvironment" title="Permalink to this definition">¶</a></dt>
    <dd><p>Bases: <a class="reference internal" href="#flow.envs.loop_accel.SimpleAccelerationEnvironment" title="flow.envs.loop_accel.SimpleAccelerationEnvironment"><code class="xref py py-class docutils literal"><span class="pre">flow.envs.loop_accel.SimpleAccelerationEnvironment</span></code></a></p>
    <p>This environment is an extension of the SimpleAccelerationEnvironment, with
    the exception that only local information is provided to the agent about the
    network; i.e. headway, velocity, and velocity difference. The reward
    function, however, continues to reward global network performance.</p>
    <p>NOTE: The environment also assumes that there is only one autonomous vehicle
    is in the network.</p>
    <dl class="method">
    <dt id="flow.envs.loop_accel.SimplePartiallyObservableEnvironment.get_state">
    <code class="descname">get_state</code><span class="sig-paren">(</span><em>**kwargs</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/loop_accel.html#SimplePartiallyObservableEnvironment.get_state"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.loop_accel.SimplePartiallyObservableEnvironment.get_state" title="Permalink to this definition">¶</a></dt>
    <dd><p>See parent class</p>
    <p>The state is an array consisting of the speed of the rl vehicle, the
    relative speed of the vehicle ahead of it, and the headway between the
    rl vehicle and the vehicle ahead of it.</p>
    </dd></dl>

    <dl class="attribute">
    <dt id="flow.envs.loop_accel.SimplePartiallyObservableEnvironment.observation_space">
    <code class="descname">observation_space</code><a class="headerlink" href="#flow.envs.loop_accel.SimplePartiallyObservableEnvironment.observation_space" title="Permalink to this definition">¶</a></dt>
    <dd><p>See parent class</p>
    </dd></dl>

    </dd></dl>

    </div>
    <div class="section" id="module-flow.envs.loop_merges">
    <span id="flow-envs-loop-merges-module"></span><h2>flow.envs.loop_merges module<a class="headerlink" href="#module-flow.envs.loop_merges" title="Permalink to this headline">¶</a></h2>
    <dl class="class">
    <dt id="flow.envs.loop_merges.SimpleLoopMergesEnvironment">
    <em class="property">class </em><code class="descclassname">flow.envs.loop_merges.</code><code class="descname">SimpleLoopMergesEnvironment</code><span class="sig-paren">(</span><em>env_params</em>, <em>sumo_params</em>, <em>scenario</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/loop_merges.html#SimpleLoopMergesEnvironment"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.loop_merges.SimpleLoopMergesEnvironment" title="Permalink to this definition">¶</a></dt>
    <dd><p>Bases: <a class="reference internal" href="#flow.envs.base_env.SumoEnvironment" title="flow.envs.base_env.SumoEnvironment"><code class="xref py py-class docutils literal"><span class="pre">flow.envs.base_env.SumoEnvironment</span></code></a></p>
    <p>Fully functional environment. Takes in an <em>acceleration</em> as an action.
    Reward function is negative norm of the difference between the velocities of
    each vehicle, and the target velocity. State function is a vector of the
    velocities, positions, and edge IDs for each vehicle.</p>
    <dl class="attribute">
    <dt id="flow.envs.loop_merges.SimpleLoopMergesEnvironment.action_space">
    <code class="descname">action_space</code><a class="headerlink" href="#flow.envs.loop_merges.SimpleLoopMergesEnvironment.action_space" title="Permalink to this definition">¶</a></dt>
    <dd><p>See parent class</p>
    </dd></dl>

    <dl class="method">
    <dt id="flow.envs.loop_merges.SimpleLoopMergesEnvironment.additional_command">
    <code class="descname">additional_command</code><span class="sig-paren">(</span><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/loop_merges.html#SimpleLoopMergesEnvironment.additional_command"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.loop_merges.SimpleLoopMergesEnvironment.additional_command" title="Permalink to this definition">¶</a></dt>
    <dd><p>Vehicles that are meant to stay in the ring are rerouted whenever they
    reach a new edge.</p>
    </dd></dl>

    <dl class="method">
    <dt id="flow.envs.loop_merges.SimpleLoopMergesEnvironment.apply_acceleration">
    <code class="descname">apply_acceleration</code><span class="sig-paren">(</span><em>veh_ids</em>, <em>acc</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/loop_merges.html#SimpleLoopMergesEnvironment.apply_acceleration"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.loop_merges.SimpleLoopMergesEnvironment.apply_acceleration" title="Permalink to this definition">¶</a></dt>
    <dd><p>See parent class.</p>
    <p>In addition, merging vehicles travel at the target velocity at the
    merging lanes, and vehicle that are about to leave the network stop.</p>
    </dd></dl>

    <dl class="method">
    <dt id="flow.envs.loop_merges.SimpleLoopMergesEnvironment.apply_rl_actions">
    <code class="descname">apply_rl_actions</code><span class="sig-paren">(</span><em>rl_actions</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/loop_merges.html#SimpleLoopMergesEnvironment.apply_rl_actions"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.loop_merges.SimpleLoopMergesEnvironment.apply_rl_actions" title="Permalink to this definition">¶</a></dt>
    <dd><p>See parent class</p>
    </dd></dl>

    <dl class="method">
    <dt id="flow.envs.loop_merges.SimpleLoopMergesEnvironment.compute_reward">
    <code class="descname">compute_reward</code><span class="sig-paren">(</span><em>state</em>, <em>rl_actions</em>, <em>**kwargs</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/loop_merges.html#SimpleLoopMergesEnvironment.compute_reward"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.loop_merges.SimpleLoopMergesEnvironment.compute_reward" title="Permalink to this definition">¶</a></dt>
    <dd><p>See parent class</p>
    </dd></dl>

    <dl class="method">
    <dt id="flow.envs.loop_merges.SimpleLoopMergesEnvironment.get_state">
    <code class="descname">get_state</code><span class="sig-paren">(</span><em>**kwargs</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/loop_merges.html#SimpleLoopMergesEnvironment.get_state"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.loop_merges.SimpleLoopMergesEnvironment.get_state" title="Permalink to this definition">¶</a></dt>
    <dd><p>See parent class</p>
    </dd></dl>

    <dl class="attribute">
    <dt id="flow.envs.loop_merges.SimpleLoopMergesEnvironment.observation_space">
    <code class="descname">observation_space</code><a class="headerlink" href="#flow.envs.loop_merges.SimpleLoopMergesEnvironment.observation_space" title="Permalink to this definition">¶</a></dt>
    <dd><p>See parent class</p>
    </dd></dl>

    <dl class="method">
    <dt id="flow.envs.loop_merges.SimpleLoopMergesEnvironment.sort_by_position">
    <code class="descname">sort_by_position</code><span class="sig-paren">(</span><em>**kwargs</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/loop_merges.html#SimpleLoopMergesEnvironment.sort_by_position"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.loop_merges.SimpleLoopMergesEnvironment.sort_by_position" title="Permalink to this definition">¶</a></dt>
    <dd><p>See parent class.</p>
    <p>Vehicles in the ring are sorted by their relative position in the ring,
    while vehicles outside the ring are sorted according to their position
    of their respective edge.</p>
    <p>Vehicles are sorted by position on the ring, then the in-merge, and
    finally the out-merge.</p>
    </dd></dl>

    </dd></dl>

    </div>
    <div class="section" id="module-flow.envs.loop_with_perturbation">
    <span id="flow-envs-loop-with-perturbation-module"></span><h2>flow.envs.loop_with_perturbation module<a class="headerlink" href="#module-flow.envs.loop_with_perturbation" title="Permalink to this headline">¶</a></h2>
    <dl class="class">
    <dt id="flow.envs.loop_with_perturbation.PerturbationAccelerationLoop">
    <em class="property">class </em><code class="descclassname">flow.envs.loop_with_perturbation.</code><code class="descname">PerturbationAccelerationLoop</code><span class="sig-paren">(</span><em>env_params</em>, <em>sumo_params</em>, <em>scenario</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/loop_with_perturbation.html#PerturbationAccelerationLoop"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.loop_with_perturbation.PerturbationAccelerationLoop" title="Permalink to this definition">¶</a></dt>
    <dd><p>Bases: <a class="reference internal" href="#flow.envs.loop_accel.SimpleAccelerationEnvironment" title="flow.envs.loop_accel.SimpleAccelerationEnvironment"><code class="xref py py-class docutils literal"><span class="pre">flow.envs.loop_accel.SimpleAccelerationEnvironment</span></code></a></p>
    <dl class="method">
    <dt id="flow.envs.loop_with_perturbation.PerturbationAccelerationLoop.additional_command">
    <code class="descname">additional_command</code><span class="sig-paren">(</span><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/loop_with_perturbation.html#PerturbationAccelerationLoop.additional_command"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.loop_with_perturbation.PerturbationAccelerationLoop.additional_command" title="Permalink to this definition">¶</a></dt>
    <dd></dd></dl>

    </dd></dl>

    </div>
    <div class="section" id="module-flow.envs.two_intersection">
    <span id="flow-envs-two-intersection-module"></span><h2>flow.envs.two_intersection module<a class="headerlink" href="#module-flow.envs.two_intersection" title="Permalink to this headline">¶</a></h2>
    <dl class="class">
    <dt id="flow.envs.two_intersection.TwoIntersectionEnvironment">
    <em class="property">class </em><code class="descclassname">flow.envs.two_intersection.</code><code class="descname">TwoIntersectionEnvironment</code><span class="sig-paren">(</span><em>env_params</em>, <em>sumo_params</em>, <em>scenario</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/two_intersection.html#TwoIntersectionEnvironment"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.two_intersection.TwoIntersectionEnvironment" title="Permalink to this definition">¶</a></dt>
    <dd><p>Bases: <a class="reference internal" href="#flow.envs.intersection_env.IntersectionEnvironment" title="flow.envs.intersection_env.IntersectionEnvironment"><code class="xref py py-class docutils literal"><span class="pre">flow.envs.intersection_env.IntersectionEnvironment</span></code></a></p>
    <p>Fully functional environment. Takes in an <em>acceleration</em> as an action. Reward function is negative norm of the
    difference between the velocities of each vehicle, and the target velocity. State function is a vector of the
    velocities for each vehicle.</p>
    <dl class="attribute">
    <dt id="flow.envs.two_intersection.TwoIntersectionEnvironment.action_space">
    <code class="descname">action_space</code><a class="headerlink" href="#flow.envs.two_intersection.TwoIntersectionEnvironment.action_space" title="Permalink to this definition">¶</a></dt>
    <dd><p>Actions are a set of accelerations from 0 to 15m/s
    :return:</p>
    </dd></dl>

    <dl class="method">
    <dt id="flow.envs.two_intersection.TwoIntersectionEnvironment.apply_rl_actions">
    <code class="descname">apply_rl_actions</code><span class="sig-paren">(</span><em>rl_actions</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/two_intersection.html#TwoIntersectionEnvironment.apply_rl_actions"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.two_intersection.TwoIntersectionEnvironment.apply_rl_actions" title="Permalink to this definition">¶</a></dt>
    <dd><p>See parent class</p>
    </dd></dl>

    <dl class="method">
    <dt id="flow.envs.two_intersection.TwoIntersectionEnvironment.compute_reward">
    <code class="descname">compute_reward</code><span class="sig-paren">(</span><em>state</em>, <em>rl_actions</em>, <em>**kwargs</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/two_intersection.html#TwoIntersectionEnvironment.compute_reward"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.two_intersection.TwoIntersectionEnvironment.compute_reward" title="Permalink to this definition">¶</a></dt>
    <dd><p>See parent class</p>
    </dd></dl>

    <dl class="method">
    <dt id="flow.envs.two_intersection.TwoIntersectionEnvironment.get_state">
    <code class="descname">get_state</code><span class="sig-paren">(</span><em>**kwargs</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/two_intersection.html#TwoIntersectionEnvironment.get_state"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.two_intersection.TwoIntersectionEnvironment.get_state" title="Permalink to this definition">¶</a></dt>
    <dd><p>See parent class
    The state is an array the velocities for each vehicle
    :return: a matrix of velocities and absolute positions for each vehicle</p>
    </dd></dl>

    <dl class="attribute">
    <dt id="flow.envs.two_intersection.TwoIntersectionEnvironment.observation_space">
    <code class="descname">observation_space</code><a class="headerlink" href="#flow.envs.two_intersection.TwoIntersectionEnvironment.observation_space" title="Permalink to this definition">¶</a></dt>
    <dd><p>See parent class
    An observation is an array the velocities for each vehicle</p>
    </dd></dl>

    </dd></dl>

    </div>
    <div class="section" id="module-flow.envs.two_loops_one_merging">
    <span id="flow-envs-two-loops-one-merging-module"></span><h2>flow.envs.two_loops_one_merging module<a class="headerlink" href="#module-flow.envs.two_loops_one_merging" title="Permalink to this headline">¶</a></h2>
    <dl class="class">
    <dt id="flow.envs.two_loops_one_merging.TwoLoopsOneMergingEnvironment">
    <em class="property">class </em><code class="descclassname">flow.envs.two_loops_one_merging.</code><code class="descname">TwoLoopsOneMergingEnvironment</code><span class="sig-paren">(</span><em>env_params</em>, <em>sumo_params</em>, <em>scenario</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/two_loops_one_merging.html#TwoLoopsOneMergingEnvironment"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.two_loops_one_merging.TwoLoopsOneMergingEnvironment" title="Permalink to this definition">¶</a></dt>
    <dd><p>Bases: <a class="reference internal" href="#flow.envs.base_env.SumoEnvironment" title="flow.envs.base_env.SumoEnvironment"><code class="xref py py-class docutils literal"><span class="pre">flow.envs.base_env.SumoEnvironment</span></code></a></p>
    <p>Fully functional environment. Differs from the SimpleAccelerationEnvironment
    in loop_accel in that vehicles in this environment may follow one of two
    routes (continuously on the smaller ring or merging in and out of the
    smaller ring). Accordingly, the single global reference for position is
    replaced with a reference in each ring.</p>
    <dl class="attribute">
    <dt id="flow.envs.two_loops_one_merging.TwoLoopsOneMergingEnvironment.action_space">
    <code class="descname">action_space</code><a class="headerlink" href="#flow.envs.two_loops_one_merging.TwoLoopsOneMergingEnvironment.action_space" title="Permalink to this definition">¶</a></dt>
    <dd><p>See parent class.</p>
    <p>Actions are a set of accelerations from max-deacc to max-acc for each
    rl vehicle.</p>
    </dd></dl>

    <dl class="method">
    <dt id="flow.envs.two_loops_one_merging.TwoLoopsOneMergingEnvironment.apply_rl_actions">
    <code class="descname">apply_rl_actions</code><span class="sig-paren">(</span><em>rl_actions</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/two_loops_one_merging.html#TwoLoopsOneMergingEnvironment.apply_rl_actions"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.two_loops_one_merging.TwoLoopsOneMergingEnvironment.apply_rl_actions" title="Permalink to this definition">¶</a></dt>
    <dd><p>See parent class.</p>
    </dd></dl>

    <dl class="method">
    <dt id="flow.envs.two_loops_one_merging.TwoLoopsOneMergingEnvironment.compute_reward">
    <code class="descname">compute_reward</code><span class="sig-paren">(</span><em>state</em>, <em>rl_actions</em>, <em>**kwargs</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/two_loops_one_merging.html#TwoLoopsOneMergingEnvironment.compute_reward"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.two_loops_one_merging.TwoLoopsOneMergingEnvironment.compute_reward" title="Permalink to this definition">¶</a></dt>
    <dd><p>See parent class</p>
    </dd></dl>

    <dl class="method">
    <dt id="flow.envs.two_loops_one_merging.TwoLoopsOneMergingEnvironment.get_state">
    <code class="descname">get_state</code><span class="sig-paren">(</span><em>**kwargs</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/two_loops_one_merging.html#TwoLoopsOneMergingEnvironment.get_state"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.two_loops_one_merging.TwoLoopsOneMergingEnvironment.get_state" title="Permalink to this definition">¶</a></dt>
    <dd><p>See parent class.</p>
    <p>The state is an array the velocities, edge counts, and relative
    positions on the edge, for each vehicle.</p>
    </dd></dl>

    <dl class="attribute">
    <dt id="flow.envs.two_loops_one_merging.TwoLoopsOneMergingEnvironment.observation_space">
    <code class="descname">observation_space</code><a class="headerlink" href="#flow.envs.two_loops_one_merging.TwoLoopsOneMergingEnvironment.observation_space" title="Permalink to this definition">¶</a></dt>
    <dd><p>See parent class.</p>
    <p>An observation is an array the velocities, positions, and edges for
    each vehicle</p>
    </dd></dl>

    <dl class="method">
    <dt id="flow.envs.two_loops_one_merging.TwoLoopsOneMergingEnvironment.sort_by_position">
    <code class="descname">sort_by_position</code><span class="sig-paren">(</span><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/envs/two_loops_one_merging.html#TwoLoopsOneMergingEnvironment.sort_by_position"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.envs.two_loops_one_merging.TwoLoopsOneMergingEnvironment.sort_by_position" title="Permalink to this definition">¶</a></dt>
    <dd><p>See parent class</p>
    <p>Instead of being sorted by a global reference, vehicles in this
    environment are sorted with regards to which ring this currently
    reside on.</p>
    </dd></dl>

    </dd></dl>

    </div>

Module contents
---------------

.. automodule:: flow.envs
    :members:
    :undoc-members:
    :show-inheritance:
