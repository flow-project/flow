Visualization
*******************

Flow supports visualization of both rllab and RLlib computational experiments.
When using one of the below visualizers, a window will appear similar to the
one in the figure below. Click on the play button (highlighted in red) and the
simulation will begin, with the autonomous vehicles exhibiting the behavior
trained by the reinforcement learning algorithm.

.. image:: ../img/visualizing.png
   :width: 400
   :align: center

rllab
=====
Call the rllab visualizer with
:: 

	python ./visualizer_rllab.py /result_dir/itr_XXX.pkl

The rllab visualizer also takes some inputs:

- ``--num_rollouts``
- ``--plotname``
- ``--use_sumogui``
- ``--run_long``
- ``--emission_to_csv``

The ``params.pkl`` file can be used as well.



RLlib
=====
Call the RLlib visualizer with
::

    python ./visualizer_rllib.py /ray_results/result_dir 1

The first command-line argument corresponds to the directory containing 
experiment results (usually within RLlib's ``ray_results``). The second is 
the checkpoint number, corresponding to the iteration number you wish to 
visualize. 

Parameter storage
-----------------
RLlib doesn't automatically store all parameters needed for restoring the 
state of a Flow experiment upon visualization. As such, Flow experiments in RLlib
include code to store relevant parameters. Include the following code snippet in
RLlib experiments you will need to visualize
::

    # Logging out flow_params to ray's experiment result folder
    from flow.utils.rllib import FlowParamsEncoder
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

This saves the relevant flow parameters to recreate Flow and perform
a rollout.

Another thing to keep in mind is that Flow parameters in RLlib experiments
should be defined **outside** of the ``make_create_env`` function. This allows
that environment creator function to use other experiment parameters later,
upon visualization. 
