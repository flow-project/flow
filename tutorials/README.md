# Flow Tutorials

## Setup

1. Make sure you have Python 3 installed (we recommend using the [Anaconda
   Python distribution](https://www.continuum.io/downloads)).
2. **Install Jupyter** with `pip install jupyter`. Verify that you can start
   a Jupyter notebook with the command `jupyter-notebook`.
3. **Install Flow** by executing the following [installation instructions](
   https://flow.readthedocs.io/en/latest/flow_setup.html).

## Tutorials

Each file ``tutorials/tutorial*.ipynb`` is a separate tutorial. They can be
opened in a Jupyter notebook by running the following commands:

```shell
source activate flow
cd <flow-path>/tutorials
jupyter-notebook
```

Instructions are written in each file. To do each tutorial, first run all of
the cells in the Jupyter notebook. Then modify the ones that need to be
modified in order to prevent any exceptions from being raised. Throughout these
tutorials, you may find the
[Flow documentation](https://flow.readthedocs.io/en/latest/) helpful.

> **Common error:** if, when running a notebook, you run into an error of the form
> `ImportError: No module named flow.something`, this probably means that the
> `flow` Conda environment is not active in your notebook. Go into the
> [Conda tab](https://stackoverflow.com/questions/38984238/how-to-set-a-default-environment-for-anaconda-jupyter)
> and make sure that `flow` is selected. In case you don't have this Conda tab,
> try running `conda install nb_conda` just after `source activate flow`,
> then open the notebook again. If this doesn't work either, you can try other
> solutions [here](https://stackoverflow.com/questions/39604271/conda-environments-not-showing-up-in-jupyter-notebook)
> , or you can launch a Jupyter notebook using the `flow` environment directly
> from the [Anaconda Navigator](https://docs.anaconda.com/anaconda/navigator/).
> If you have the conda tab but you still get the error, open a new terminal
> and execute the following commands:
> ```
> cd /
> source activate flow
> python
> ```
> (`cd /` is to make sure that `flow` is not in the folder you run `python` from). Then, in the Python interface that opens, run `import flow`. If you get an `ImportError`, this means you haven't installed Flow in your environment. Go back to the [installation instructions](https://flow.readthedocs.io/en/latest/flow_setup.html), especially the part where you do `pip install -e .` after having done `source activate flow`.

The content of each tutorial is as follows:

**Tutorial 0:** High-level introduction to Flow.

**Tutorial 1:** Running SUMO simulations in Flow.

**Tutorial 2:** Running Aimsun simulations in Flow.

**Tutorial 3:** Running RLlib experiments for mixed-autonomy traffic.

**Tutorial 4:** Saving and visualizing resuls from non-RL simulations and
testing simulations in the presence of an RLlib agent.

**Tutorial 5:** Creating custom networks.

**Tutorial 6:** Importing networks from OpenStreetMap.

**Tutorial 7:** Importing networks from simulator-specific template files.

**Tutorial 8:** Creating custom environments.

**Tutorial 9:** Creating custom controllers.

**Tutorial 10:** Traffic lights.

**Tutorial 11:** Running simulations with inflows of vehicles.

**Tutorial 12:** Running the bottleneck environment.

**Tutorial 13:** Running rllib experiments on Amazon EC2 instances
