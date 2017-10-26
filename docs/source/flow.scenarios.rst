flow.scenarios
==============

.. raw:: html

	<div class="section" id="flow-scenarios-package">
	<div class="section" id="subpackages">
	<h2>Subpackages<a class="headerlink" href="#subpackages" title="Permalink to this headline">¶</a></h2>
	<div class="toctree-wrapper compound">
	<ul>
	<li class="toctree-l1"><a class="reference internal" href="flow.scenarios.loop_merges.html">flow.scenarios.loop_merges</a><ul>
	<li class="toctree-l2"><a class="reference internal" href="flow.scenarios.loop_merges.html#submodules">Submodules</a></li>
	<li class="toctree-l2"><a class="reference internal" href="flow.scenarios.loop_merges.html#module-flow.scenarios.loop_merges.gen">flow.scenarios.loop_merges.gen module</a></li>
	<li class="toctree-l2"><a class="reference internal" href="flow.scenarios.loop_merges.html#module-flow.scenarios.loop_merges.loop_merges_scenario">flow.scenarios.loop_merges.loop_merges_scenario module</a></li>
	<li class="toctree-l2"><a class="reference internal" href="flow.scenarios.loop_merges.html#module-flow.scenarios.loop_merges">Module contents</a></li>
	</ul>
	</li>
	</ul>
	</div>
	</div>
	<div class="section" id="submodules">
	<h2>Submodules<a class="headerlink" href="#submodules" title="Permalink to this headline">¶</a></h2>
	</div>
	<div class="section" id="module-flow.scenarios.base_scenario">
	<span id="flow-scenarios-base-scenario-module"></span><h2>flow.scenarios.base_scenario module<a class="headerlink" href="#module-flow.scenarios.base_scenario" title="Permalink to this headline">¶</a></h2>
	<dl class="class">
	<dt id="flow.scenarios.base_scenario.Scenario">
	<em class="property">class </em><code class="descclassname">flow.scenarios.base_scenario.</code><code class="descname">Scenario</code><span class="sig-paren">(</span><em>name</em>, <em>generator_class</em>, <em>vehicles</em>, <em>net_params</em>, <em>initial_config=&lt;flow.core.params.InitialConfig object&gt;</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/scenarios/base_scenario.html#Scenario"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.scenarios.base_scenario.Scenario" title="Permalink to this definition">¶</a></dt>
	<dd><p>Bases: <code class="xref py py-class docutils literal"><span class="pre">rllab.core.serializable.Serializable</span></code></p>
	<dl class="method">
	<dt id="flow.scenarios.base_scenario.Scenario.gen_custom_start_pos">
	<code class="descname">gen_custom_start_pos</code><span class="sig-paren">(</span><em>initial_config</em>, <em>**kwargs</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/scenarios/base_scenario.html#Scenario.gen_custom_start_pos"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.scenarios.base_scenario.Scenario.gen_custom_start_pos" title="Permalink to this definition">¶</a></dt>
	<dd><p>Generates a user defined set of starting postions. Optional</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><ul class="first simple">
	<li><strong>initial_config</strong> (<em>InitialConfig type</em>) – see flow/core/params.py</li>
	<li><strong>kwargs</strong> (<em>dict</em>) – extra components, usually defined during reset to overwrite initial
	config parameters</li>
	</ul>
	</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Returns:</th><td class="field-body"><p class="first last"><ul class="simple">
	<li><strong>startpositions</strong> (<em>list</em>) – list of start positions [(edge0, pos0), (edge1, pos1), …]</li>
	<li><strong>startlanes</strong> (<em>list</em>) – list of start lanes</li>
	</ul>
	</p>
	</td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="method">
	<dt id="flow.scenarios.base_scenario.Scenario.gen_even_start_pos">
	<code class="descname">gen_even_start_pos</code><span class="sig-paren">(</span><em>initial_config</em>, <em>**kwargs</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/scenarios/base_scenario.html#Scenario.gen_even_start_pos"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.scenarios.base_scenario.Scenario.gen_even_start_pos" title="Permalink to this definition">¶</a></dt>
	<dd><p>Generates start positions that are uniformly spaced across the network.</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><ul class="first simple">
	<li><strong>initial_config</strong> (<em>InitialConfig type</em>) – see flow/core/params.py</li>
	<li><strong>kwargs</strong> (<em>dict</em>) – extra components, usually defined during reset to overwrite initial
	config parameters</li>
	</ul>
	</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Returns:</th><td class="field-body"><p class="first last"><ul class="simple">
	<li><strong>startpositions</strong> (<em>list</em>) – list of start positions [(edge0, pos0), (edge1, pos1), …]</li>
	<li><strong>startlanes</strong> (<em>list</em>) – list of start lanes</li>
	</ul>
	</p>
	</td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="method">
	<dt id="flow.scenarios.base_scenario.Scenario.gen_gaussian_additive_start_pos">
	<code class="descname">gen_gaussian_additive_start_pos</code><span class="sig-paren">(</span><em>initial_config</em>, <em>**kwargs</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/scenarios/base_scenario.html#Scenario.gen_gaussian_additive_start_pos"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.scenarios.base_scenario.Scenario.gen_gaussian_additive_start_pos" title="Permalink to this definition">¶</a></dt>
	<dd><p>Generate random start positions via additive Gaussian.
	WARNING: this does not absolutely gaurantee that the order of
	vehicles is preserved.</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><ul class="first simple">
	<li><strong>initial_config</strong> (<em>InitialConfig type</em>) – see flow/core/params.py</li>
	<li><strong>kwargs</strong> (<em>dict</em>) – extra components, usually defined during reset to overwrite initial
	config parameters</li>
	</ul>
	</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Returns:</th><td class="field-body"><p class="first last"><ul class="simple">
	<li><strong>startpositions</strong> (<em>list</em>) – list of start positions [(edge0, pos0), (edge1, pos1), …]</li>
	<li><strong>startlanes</strong> (<em>list</em>) – list of start lanes</li>
	</ul>
	</p>
	</td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="method">
	<dt id="flow.scenarios.base_scenario.Scenario.gen_gaussian_start_pos">
	<code class="descname">gen_gaussian_start_pos</code><span class="sig-paren">(</span><em>initial_config</em>, <em>**kwargs</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/scenarios/base_scenario.html#Scenario.gen_gaussian_start_pos"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.scenarios.base_scenario.Scenario.gen_gaussian_start_pos" title="Permalink to this definition">¶</a></dt>
	<dd><p>Generates start positions that are perturbed from a uniformly spaced
	distribution by some gaussian noise.</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><ul class="first simple">
	<li><strong>initial_config</strong> (<em>InitialConfig type</em>) – see flow/core/params.py</li>
	<li><strong>kwargs</strong> (<em>dict</em>) – extra components, usually defined during reset to overwrite initial
	config parameters</li>
	</ul>
	</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Returns:</th><td class="field-body"><p class="first last"><ul class="simple">
	<li><strong>startpositions</strong> (<em>list</em>) – list of start positions [(edge0, pos0), (edge1, pos1), …]</li>
	<li><strong>startlanes</strong> (<em>list</em>) – list of start lanes</li>
	</ul>
	</p>
	</td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="method">
	<dt id="flow.scenarios.base_scenario.Scenario.generate">
	<code class="descname">generate</code><span class="sig-paren">(</span><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/scenarios/base_scenario.html#Scenario.generate"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.scenarios.base_scenario.Scenario.generate" title="Permalink to this definition">¶</a></dt>
	<dd><p>Applies self.generator_class to create a net and corresponding cfg
	files, including placement of vehicles (name.rou.xml).</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Returns:</th><td class="field-body"><strong>cfg</strong> – path to configuration (.sumo.cfg) file</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Return type:</th><td class="field-body">str</td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="method">
	<dt id="flow.scenarios.base_scenario.Scenario.generate_starting_positions">
	<code class="descname">generate_starting_positions</code><span class="sig-paren">(</span><em>**kwargs</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/scenarios/base_scenario.html#Scenario.generate_starting_positions"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.scenarios.base_scenario.Scenario.generate_starting_positions" title="Permalink to this definition">¶</a></dt>
	<dd><p>Generates starting positions for vehicles in the network. Calls all
	other starting position generating classes.</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><strong>kwargs</strong> (<em>dict</em>) – additional arguments that may be updated beyond initial
	configurations, such as modifying the starting position</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Returns:</th><td class="field-body"><ul class="simple">
	<li><strong>startpositions</strong> (<em>list</em>) – list of start positions [(edge0, pos0), (edge1, pos1), …]</li>
	<li><strong>startlanes</strong> (<em>list</em>) – list of start lanes</li>
	</ul>
	</td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="method">
	<dt id="flow.scenarios.base_scenario.Scenario.get_edge">
	<code class="descname">get_edge</code><span class="sig-paren">(</span><em>x</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/scenarios/base_scenario.html#Scenario.get_edge"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.scenarios.base_scenario.Scenario.get_edge" title="Permalink to this definition">¶</a></dt>
	<dd><p>Given an absolute position x on the track, returns the edge (name) and
	relative position on that edge.</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><strong>x</strong> (<em>float</em>) – absolute position in network</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Returns:</th><td class="field-body"><strong>edge position</strong> – 1st element: edge name (such as bottom, right, etc.)
	2nd element: relative position on edge</td>
	</tr>
	<tr class="field-odd field"><th class="field-name">Return type:</th><td class="field-body">tup</td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="method">
	<dt id="flow.scenarios.base_scenario.Scenario.get_x">
	<code class="descname">get_x</code><span class="sig-paren">(</span><em>edge</em>, <em>position</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/scenarios/base_scenario.html#Scenario.get_x"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.scenarios.base_scenario.Scenario.get_x" title="Permalink to this definition">¶</a></dt>
	<dd><p>Given an edge name and relative position, return the absolute position
	on the track.</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><ul class="first simple">
	<li><strong>edge</strong> (<em>str</em>) – name of the edge</li>
	<li><strong>position</strong> (<em>float</em>) – relative position on the edge</li>
	</ul>
	</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Returns:</th><td class="field-body"><p class="first"><strong>absolute_position</strong> – position with respect to some global reference</p>
	</td>
	</tr>
	<tr class="field-odd field"><th class="field-name">Return type:</th><td class="field-body"><p class="first last">float</p>
	</td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="method">
	<dt id="flow.scenarios.base_scenario.Scenario.specify_edge_starts">
	<code class="descname">specify_edge_starts</code><span class="sig-paren">(</span><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/scenarios/base_scenario.html#Scenario.specify_edge_starts"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.scenarios.base_scenario.Scenario.specify_edge_starts" title="Permalink to this definition">¶</a></dt>
	<dd><p>Defines edge starts for road sections with respect to some global
	reference frame.
	MUST BE implemented in any new scenario subclass.</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Returns:</th><td class="field-body"><strong>edgestarts</strong> – list of edge names and starting positions,
	ex: [(edge0, pos0), (edge1, pos1), …]</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Return type:</th><td class="field-body">list</td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="method">
	<dt id="flow.scenarios.base_scenario.Scenario.specify_internal_edge_starts">
	<code class="descname">specify_internal_edge_starts</code><span class="sig-paren">(</span><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/scenarios/base_scenario.html#Scenario.specify_internal_edge_starts"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.scenarios.base_scenario.Scenario.specify_internal_edge_starts" title="Permalink to this definition">¶</a></dt>
	<dd><p>Defines the edge starts for internal edge nodes (caused by finite-length
	connections between road sections) with respect to some global reference
	frame. Does not need to be specified if “no-internal-links” is set to
	True in net_params.</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Returns:</th><td class="field-body"><strong>internal_edgestarts</strong> – list of internal junction names and starting positions,
	ex: [(internal0, pos0), (internal1, pos1), …]</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Return type:</th><td class="field-body">list</td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="method">
	<dt id="flow.scenarios.base_scenario.Scenario.specify_intersection_edge_starts">
	<code class="descname">specify_intersection_edge_starts</code><span class="sig-paren">(</span><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/scenarios/base_scenario.html#Scenario.specify_intersection_edge_starts"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.scenarios.base_scenario.Scenario.specify_intersection_edge_starts" title="Permalink to this definition">¶</a></dt>
	<dd><p>Defines edge starts for intersections with respect to some global
	reference frame. Need not be specified if no intersections exist.
	These values can be used to determine the distance of some agent from
	the nearest and/or all intersections.</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Returns:</th><td class="field-body"><strong>intersection_edgestarts</strong> – list of intersection names and starting positions,
	ex: [(intersection0, pos0), (intersection1, pos1), …]</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Return type:</th><td class="field-body">list</td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	</dd></dl>

	</div>


Module contents
---------------

.. automodule:: flow.scenarios
    :members:
    :undoc-members:
    :show-inheritance:
