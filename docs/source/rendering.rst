Flow Renderer
*******************

Flow has a customized renderer built on pyglet.  It provides a convenient
interface for image-based learning. It supports rendering the simulation as a
top-view snapshot and can extract local observations of RL vehicles or
tracked human vehicles.

An example of the minicity rendered by the pyglet renderer can found below.

.. image:: ../img/minicity_pyglet_render.png
   :width: 500
   :align: center

The green arrows are untracked human vehicles, while the blue arrows are RL
vehicles and tracked human vehicles. Color saturation is proportional to
speed. For example, the greener the human vehicles, the faster they drive.
The observation radius of RL and tracked human vehicles are marked by circles.

An example of an extracted local observation is shown as follows.

.. image:: ../img/local_obs_pyglet_render.png
   :width: 250
   :align: center

To generate and save the rendering seen above, use
::
    exp = minicity_example(render='drgb',  # Render in dynamic RGB colors
                           save_render=True,  # Save rendering
                           sight_radius=30,  # Radius of obs.
                           pxpm=3,  # Render at 3 pixel per meter
                           show_radius=True)  # Show obs. radius

The rendered frames will be save at ``~/flow_rendering``. For more information,
check the `PygletRenderer <https://github.com/flow-project/flow/blob/master/flow/renderer/pyglet_renderer.py>`_ class.

*The custom renderer is slower than SUMO's built-in GUI. We are working on
performance optimization and will update a faster version in near future.*
