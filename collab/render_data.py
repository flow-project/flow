import numpy as np
import sys
from flow.renderer.pyglet_renderer import PygletRenderer as Renderer
import time

"""
Example usage: python render_data.py /path/to/data.npy
"""

path = sys.argv[1]
print("Rendering " + path + "...")
data = np.load(path); print(data.shape)

network = data[0]
data = data[2:]
mode = "drgb"
save_global = True
save_local = not save_global
sight_radius = 25
pxpm = 5
show_radius = True
renderer = Renderer(
    network,
    mode,
    save_render=True,
    sight_radius=sight_radius,
    pxpm=pxpm,
    show_radius=show_radius)

for _data in data:
    human_orientations, machine_orientations,\
    human_dynamics, machine_dynamics,\
    human_logs, machine_logs = _data
    human_dynamics = np.asarray(human_dynamics)*5
    machine_dynamics = np.asarray(machine_dynamics)*5
    frame = renderer.render(human_orientations,
                            machine_orientations,
                            human_dynamics,
                            machine_dynamics,
                            human_logs,
                            machine_logs,
                            save_render=save_global)
    for orientation, log in zip(machine_orientations, machine_logs):
        id = log[-1]
        sight = renderer.get_sight(orientation, id, save_render=save_local)

renderer.close()
