.. raw:: html
	
	<div class="section" id="flow-scenarios-loop-merges-package">
	<h1>flow.scenarios.loop_merges<a class="headerlink" href="#flow-scenarios-loop-merges-package" title="Permalink to this headline">¶</a></h1>
	<div class="section" id="submodules">
	<h2>Submodules<a class="headerlink" href="#submodules" title="Permalink to this headline">¶</a></h2>
	</div>
	<div class="section" id="module-flow.scenarios.loop_merges.gen">
	<span id="flow-scenarios-loop-merges-gen-module"></span><h2>flow.scenarios.loop_merges.gen module<a class="headerlink" href="#module-flow.scenarios.loop_merges.gen" title="Permalink to this headline">¶</a></h2>
	<dl class="class">
	<dt id="flow.scenarios.loop_merges.gen.LoopMergesGenerator">
	<em class="property">class </em><code class="descclassname">flow.scenarios.loop_merges.gen.</code><code class="descname">LoopMergesGenerator</code><span class="sig-paren">(</span><em>net_params</em>, <em>base</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/scenarios/loop_merges/gen.html#LoopMergesGenerator"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.scenarios.loop_merges.gen.LoopMergesGenerator" title="Permalink to this definition">¶</a></dt>
	<dd><p>Bases: <a class="reference internal" href="flow.core.html#flow.core.generator.Generator" title="flow.core.generator.Generator"><code class="xref py py-class docutils literal"><span class="pre">flow.core.generator.Generator</span></code></a></p>
	<p>Generator for loop with merges sim. Required from net_params:
	- merge_in_length: length of the merging in lane
	- merge_out_length: length of the merging out lane. May be set to None to</p>
	<blockquote>
	<div>remove the merge-out lane.</div></blockquote>
	<ul class="simple">
	<li>merge_in_angle: angle between the horizontal line and the merge-in lane
	(in radians)</li>
	<li>merge_out_angle: angle between the horizontal line and the merge-out lane
	(in radians). MUST BE greater than the merge_in_angle</li>
	<li>ring_radius: radius of the circular portion of the network.</li>
	<li>lanes: number of lanes in the network</li>
	<li>speed: max speed of vehicles in the network</li>
	</ul>
	<dl class="method">
	<dt id="flow.scenarios.loop_merges.gen.LoopMergesGenerator.make_routes">
	<code class="descname">make_routes</code><span class="sig-paren">(</span><em>scenario</em>, <em>initial_config</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/scenarios/loop_merges/gen.html#LoopMergesGenerator.make_routes"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.scenarios.loop_merges.gen.LoopMergesGenerator.make_routes" title="Permalink to this definition">¶</a></dt>
	<dd></dd></dl>

	<dl class="method">
	<dt id="flow.scenarios.loop_merges.gen.LoopMergesGenerator.specify_edges">
	<code class="descname">specify_edges</code><span class="sig-paren">(</span><em>net_params</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/scenarios/loop_merges/gen.html#LoopMergesGenerator.specify_edges"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.scenarios.loop_merges.gen.LoopMergesGenerator.specify_edges" title="Permalink to this definition">¶</a></dt>
	<dd><p>See parent class</p>
	</dd></dl>

	<dl class="method">
	<dt id="flow.scenarios.loop_merges.gen.LoopMergesGenerator.specify_nodes">
	<code class="descname">specify_nodes</code><span class="sig-paren">(</span><em>net_params</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/scenarios/loop_merges/gen.html#LoopMergesGenerator.specify_nodes"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.scenarios.loop_merges.gen.LoopMergesGenerator.specify_nodes" title="Permalink to this definition">¶</a></dt>
	<dd><p>See parent class</p>
	</dd></dl>

	<dl class="method">
	<dt id="flow.scenarios.loop_merges.gen.LoopMergesGenerator.specify_routes">
	<code class="descname">specify_routes</code><span class="sig-paren">(</span><em>net_params</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/scenarios/loop_merges/gen.html#LoopMergesGenerator.specify_routes"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.scenarios.loop_merges.gen.LoopMergesGenerator.specify_routes" title="Permalink to this definition">¶</a></dt>
	<dd><p>See parent class</p>
	</dd></dl>

	<dl class="method">
	<dt id="flow.scenarios.loop_merges.gen.LoopMergesGenerator.specify_types">
	<code class="descname">specify_types</code><span class="sig-paren">(</span><em>net_params</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/scenarios/loop_merges/gen.html#LoopMergesGenerator.specify_types"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.scenarios.loop_merges.gen.LoopMergesGenerator.specify_types" title="Permalink to this definition">¶</a></dt>
	<dd><p>See parent class</p>
	</dd></dl>

	</dd></dl>

	</div>
	<div class="section" id="module-flow.scenarios.loop_merges.loop_merges_scenario">
	<span id="flow-scenarios-loop-merges-loop-merges-scenario-module"></span><h2>flow.scenarios.loop_merges.loop_merges_scenario module<a class="headerlink" href="#module-flow.scenarios.loop_merges.loop_merges_scenario" title="Permalink to this headline">¶</a></h2>
	<dl class="class">
	<dt id="flow.scenarios.loop_merges.loop_merges_scenario.LoopMergesScenario">
	<em class="property">class </em><code class="descclassname">flow.scenarios.loop_merges.loop_merges_scenario.</code><code class="descname">LoopMergesScenario</code><span class="sig-paren">(</span><em>name</em>, <em>generator_class</em>, <em>vehicles</em>, <em>net_params</em>, <em>initial_config=None</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/scenarios/loop_merges/loop_merges_scenario.html#LoopMergesScenario"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.scenarios.loop_merges.loop_merges_scenario.LoopMergesScenario" title="Permalink to this definition">¶</a></dt>
	<dd><p>Bases: <a class="reference internal" href="flow.scenarios.html#flow.scenarios.base_scenario.Scenario" title="flow.scenarios.base_scenario.Scenario"><code class="xref py py-class docutils literal"><span class="pre">flow.scenarios.base_scenario.Scenario</span></code></a></p>
	<dl class="method">
	<dt id="flow.scenarios.loop_merges.loop_merges_scenario.LoopMergesScenario.gen_even_start_pos">
	<code class="descname">gen_even_start_pos</code><span class="sig-paren">(</span><em>initial_config</em>, <em>**kwargs</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/scenarios/loop_merges/loop_merges_scenario.html#LoopMergesScenario.gen_even_start_pos"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.scenarios.loop_merges.loop_merges_scenario.LoopMergesScenario.gen_even_start_pos" title="Permalink to this definition">¶</a></dt>
	<dd><p>See base class</p>
	</dd></dl>

	<dl class="method">
	<dt id="flow.scenarios.loop_merges.loop_merges_scenario.LoopMergesScenario.gen_gaussian_additive_start_pos">
	<code class="descname">gen_gaussian_additive_start_pos</code><span class="sig-paren">(</span><em>initial_config</em>, <em>**kwargs</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/scenarios/loop_merges/loop_merges_scenario.html#LoopMergesScenario.gen_gaussian_additive_start_pos"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.scenarios.loop_merges.loop_merges_scenario.LoopMergesScenario.gen_gaussian_additive_start_pos" title="Permalink to this definition">¶</a></dt>
	<dd><p>Generate random start positions via additive Gaussian.</p>
	<p>WARNING: this does not absolutely gaurantee that the order of
	vehicles is preserved.</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Returns:</th><td class="field-body"><ul class="simple">
	<li><strong>startpositions</strong> (<em>list</em>) – start positions [(edge0, pos0), (edge1, pos1), …]</li>
	<li><strong>startlanes</strong> (<em>list</em>) – start lanes</li>
	</ul>
	</td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="method">
	<dt id="flow.scenarios.loop_merges.loop_merges_scenario.LoopMergesScenario.gen_gaussian_start_pos">
	<code class="descname">gen_gaussian_start_pos</code><span class="sig-paren">(</span><em>initial_config</em>, <em>**kwargs</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/scenarios/loop_merges/loop_merges_scenario.html#LoopMergesScenario.gen_gaussian_start_pos"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.scenarios.loop_merges.loop_merges_scenario.LoopMergesScenario.gen_gaussian_start_pos" title="Permalink to this definition">¶</a></dt>
	<dd><p>Generates start positions that are perturbed from a uniformly spaced
	distribution by some gaussian noise.</p>
	</dd></dl>

	<dl class="method">
	<dt id="flow.scenarios.loop_merges.loop_merges_scenario.LoopMergesScenario.specify_edge_starts">
	<code class="descname">specify_edge_starts</code><span class="sig-paren">(</span><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/scenarios/loop_merges/loop_merges_scenario.html#LoopMergesScenario.specify_edge_starts"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.scenarios.loop_merges.loop_merges_scenario.LoopMergesScenario.specify_edge_starts" title="Permalink to this definition">¶</a></dt>
	<dd><p>See parent class</p>
	</dd></dl>

	<dl class="method">
	<dt id="flow.scenarios.loop_merges.loop_merges_scenario.LoopMergesScenario.specify_internal_edge_starts">
	<code class="descname">specify_internal_edge_starts</code><span class="sig-paren">(</span><span class="sig-paren">)</span><a class="reference internal" href="_modules/flow/scenarios/loop_merges/loop_merges_scenario.html#LoopMergesScenario.specify_internal_edge_starts"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#flow.scenarios.loop_merges.loop_merges_scenario.LoopMergesScenario.specify_internal_edge_starts" title="Permalink to this definition">¶</a></dt>
	<dd><p>See parent class</p>
	</dd></dl>

	</dd></dl>

	</div>

Module contents
---------------

.. automodule:: flow.scenarios.loop_merges
    :members:
    :undoc-members:
    :show-inheritance:
