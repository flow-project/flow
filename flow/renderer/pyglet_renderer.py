"""Contains the pyglet renderer class."""

import pyglet
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
import cv2
import imutils
import os
from os.path import expanduser
import time
import copy
import warnings
HOME = expanduser("~")


class PygletRenderer(object):
    """Pyglet Renderer class.

    Provide a self-contained renderer module based on pyglet for visualization
    and pixel-based learning. To run renderer in a headless machine, use
    xvfb-run.

    Attributes
    ----------
    data : list
        A list of rendering data to be saved when save_render is set to
        True.
    mode : str or bool

        * False: no rendering
        * True: delegate rendering to sumo-gui for back-compatibility
        * "gray": static grayscale rendering, which is good for training
        * "dgray": dynamic grayscale rendering
        * "rgb": static RGB rendering
        * "drgb": dynamic RGB rendering, which is good for visualization

    save_render : bool
        Specify whether to save rendering data to disk
    path : str
        Specify where to store the rendering data
    sight_radius : int
        Set the radius of observation for RL vehicles (meter)
    show_radius : bool
        Specify whether to render the radius of RL observation
    time : int
        Rendering time that increments by one with every render() call
    lane_polys : list
        A list of road network polygons
    lane_colors : list
        A list of [r, g, b] specify colors of each lane in the network
    width : int
        Width of display window or frame
    height : int
        Height of display window or frame
    x_shift : float
        The shift substracted to the input x coordinate
    x_scale : float
        The scale multiplied to the input x coordinate
    y_shift : float
        The shift substracted to the input y coordinate
    y_scale : float
        The scale multiplied to the input y coordinate
    window : pyglet.window.Window
        A pyglet Window object used to create a display window.
    frame : numpy.array
        An array of size width x height x channel, where channel = 3 when
        rendering in rgb mode and channel = 1 when rendering in gray mode
    pxpm : int
        Specify rendering resolution (pixel / meter)
    """

    def __init__(self, network, mode,
                 save_render=False,
                 path=HOME+"/flow_rendering",
                 sight_radius=50,
                 show_radius=False,
                 pxpm=2,
                 alpha=1.0):
        """Initialize Pyglet Renderer.

        Parameters
        ----------
        network : list of list
            A list of road network polygons. Each polygon is expressed as
            a list of x and y coordinates, e.g., [x1, y1, x2, y2, ...]
        mode : str or bool

            * False: no rendering
            * True: delegate rendering to sumo-gui for back-compatibility
            * "gray": static grayscale rendering, which is good for training
            * "dgray": dynamic grayscale rendering
            * "rgb": static RGB rendering
            * "drgb": dynamic RGB rendering, which is good for visualization

        save_render : bool
            Specify whether to save rendering data to disk
        path : str
            Specify where to store the rendering data
        sight_radius : int
            Set the radius of observation for RL vehicles (meter)
        show_radius : bool
            Specify whether to render the radius of RL observation
        pxpm : int
            Specify rendering resolution (pixel / meter)
        alpha : int
            Specify opacity of the alpha channel.
            1.0 is fully opaque; 0.0 is fully transparent.
        """
        self.mode = mode
        if self.mode not in [True, False, "rgb", "drgb", "gray", "dgray"]:
            raise ValueError("Mode %s is not supported!" % self.mode)
        self.save_render = save_render
        self.path = path + '/' + time.strftime("%Y-%m-%d-%H%M%S")
        if self.save_render:
            if not os.path.exists(path):
                os.mkdir(path)
            os.mkdir(self.path)
            self.data = [network]
        self.sight_radius = sight_radius
        self.pxpm = pxpm  # Pixel per meter
        self.show_radius = show_radius
        self.alpha = alpha
        pyglet.gl.glEnable(pyglet.gl.GL_BLEND)
        pyglet.gl.glBlendFunc(
            pyglet.gl.GL_SRC_ALPHA, pyglet.gl.GL_ONE_MINUS_SRC_ALPHA)
        self.time = 0

        self.lane_polys = copy.deepcopy(network)
        lane_polys_flat = [pt for poly in network for pt in poly]

        polys_x = np.asarray(lane_polys_flat[::2])
        width = int(polys_x.max() - polys_x.min())
        shift = polys_x.min() - 2
        scale = (width - 4) / width
        self.width = (width + 2*self.sight_radius) * self.pxpm
        self.x_shift = shift - self.sight_radius
        self.x_scale = scale

        polys_y = np.asarray(lane_polys_flat[1::2])
        height = int(polys_y.max() - polys_y.min())
        shift = polys_y.min() - 2
        scale = (height - 4) / height
        self.height = (height + 2*self.sight_radius) * self.pxpm
        self.y_shift = shift - self.sight_radius
        self.y_scale = scale

        self.lane_colors = []
        for lane_poly in self.lane_polys:
            lane_poly[::2] = [(x-self.x_shift)*self.x_scale*self.pxpm
                              for x in lane_poly[::2]]
            lane_poly[1::2] = [(y-self.y_shift)*self.y_scale*self.pxpm
                               for y in lane_poly[1::2]]
            color = [c for _ in range(int(len(lane_poly)/2))
                     for c in [224, 224, 224, int(self.alpha*255)]]
            self.lane_colors.append(color)

        try:
            self.window = pyglet.window.Window(width=self.width,
                                               height=self.height)
            pyglet.gl.glClearColor(0.125, 0.125, 0.125, self.alpha)
            self.window.clear()
            self.window.switch_to()
            self.window.dispatch_events()
            self.lane_batch = pyglet.graphics.Batch()
            self._add_lane_polys()
            self.lane_batch.draw()
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            frame = np.fromstring(image_data.data, dtype=np.uint8, sep='')
            frame = frame.reshape(buffer.height, buffer.width, 4)
            self.frame = frame[::-1, :, 0:3][..., ::-1]
            self.network = self.frame.copy()
            print('Rendering with frame {} x {}...'
                  .format(self.width, self.height))
        except ImportError:
            self.window = None
            self.frame = None
            warnings.warn("Cannot access display. Aborting.", ResourceWarning)

    def render(self,
               human_orientations,
               machine_orientations,
               human_dynamics,
               machine_dynamics,
               human_logs,
               machine_logs):
        """Update the rendering frame.

        Parameters
        ----------
        human_orientations : list
            A list contains orientations of all human vehicles
            An orientation is a list contains [x, y, angle].
        machine_orientations : list
            A list contains orientations of all RL vehicles
            An orientation is a list contains [x, y, angle].
        human_dynamics : list
            A list contains the speed of all human vehicles normalized by
            max speed, i.e., speed/max_speed
            This is used to dynamically color human vehicles based on its
            velocity.
        machine_dynamics : list
            A list contains the speed of all RL vehicles normalized by
            max speed, i.e., speed/max_speed
            This is used to dynamically color RL vehicles based on its
            velocity.
        human_logs : list
            A list contains the timestep (ms), timedelta (ms), and id of
            all human vehicles
        machine_logs : list
            A list contains the timestep (ms), timedelta (ms), and id of
            all RL vehicles
        """
        if self.save_render:
            _human_orientations = copy.deepcopy(human_orientations)
            _machine_orientations = copy.deepcopy(machine_orientations)
            _human_dynamics = copy.deepcopy(human_dynamics)
            _machine_dynamics = copy.deepcopy(machine_dynamics)
            _human_logs = copy.deepcopy(human_logs)
            _machine_logs = copy.deepcopy(machine_logs)

        self.time += 1

        pyglet.gl.glClearColor(0.125, 0.125, 0.125, self.alpha)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        self.lane_batch = pyglet.graphics.Batch()
        self._add_lane_polys()
        self.lane_batch.draw()
        self.vehicle_batch = pyglet.graphics.Batch()
        if "drgb" in self.mode:
            human_cmap = self._truncate_colormap(cm.Greens, 0.2, 0.8)
            machine_cmap = self._truncate_colormap(cm.Blues, 0.2, 0.8)
            human_conditions = [
                (255*np.array(human_cmap(d)[:3]+(self.alpha,)))
                .astype(np.uint8).tolist()
                for d in human_dynamics]
            machine_conditions = [
                (255*np.array(machine_cmap(d)[:3]+(self.alpha,)))
                .astype(np.uint8).tolist()
                for d in machine_dynamics]

        elif "dgray" in self.mode:
            human_cmap = self._truncate_colormap(cm.binary, 0.55, 0.95)
            machine_cmap = self._truncate_colormap(cm.binary, 0.05, 0.45)
            human_conditions = [
                (255*np.array(human_cmap(d)[:3]+(self.alpha,)))
                .astype(np.uint8).tolist()
                for d in human_dynamics]
            machine_conditions = [
                (255*np.array(machine_cmap(d)[:3]+(self.alpha,)))
                .astype(np.uint8).tolist()
                for d in machine_dynamics]

        elif "rgb" in self.mode:
            human_conditions = [
                [0, 225, 0, int(255*self.alpha)] for d in human_dynamics]
            machine_conditions = [
                [0, 150, 200, int(255*self.alpha)] for d in machine_dynamics]

        elif "gray" in self.mode:
            human_conditions = [
                [100, 100, 100, int(255*self.alpha)] for d in human_dynamics]
            machine_conditions = [
                [150, 150, 150, int(255*self.alpha)] for d in machine_dynamics]
        else:
            raise ValueError("Unknown mode: {}".format(self.mode))

        self._add_vehicle_polys(
            human_orientations,
            human_conditions, 0
        )
        if self.show_radius:
            self._add_vehicle_polys(
                machine_orientations,
                machine_conditions, self.sight_radius
            )
        else:
            self._add_vehicle_polys(
                machine_orientations,
                machine_conditions, 0
            )
        self.vehicle_batch.draw()

        buffer = pyglet.image.get_buffer_manager().get_color_buffer()
        image_data = buffer.get_image_data()
        frame = np.fromstring(image_data.data, dtype=np.uint8, sep='')
        frame = frame.reshape(buffer.height, buffer.width, 4)
        self.frame = frame[::-1, :, 0:3][..., ::-1]
        self.window.flip()

        if self.save_render:
            cv2.imwrite("%s/frame_%06d.png" %
                        (self.path, self.time), self.frame)
            self.data.append([_human_orientations, _machine_orientations,
                              _human_dynamics, _machine_dynamics,
                              _human_logs, _machine_logs])
        if "gray" in self.mode:
            return self.frame[:, :, 0]
        else:
            return self.frame

    def close(self):
        """Terminate the renderer."""
        print('Closing renderer...')
        save_path = ''
        if self.save_render:
            save_path = '%s/data_%06d.npy' % (self.path, self.time)
            np.save(save_path, self.data)
        self.window.close()
        print('Goodbye!')
        return save_path

    def get_sight(self, orientation, veh_id):
        """Return the local observation of a vehicle.

        Parameters
        ----------
        orientation : list
            An orientation is a list contains [x, y, angle]
        veh_id : str
            The vehicle to observe for
        """
        x, y, ang = orientation
        x = (x-self.x_shift)*self.x_scale*self.pxpm
        y = (y-self.y_shift)*self.y_scale*self.pxpm
        x_med = x
        y_med = self.height - y
        sight_radius = self.sight_radius * self.pxpm
        x_min = int(x_med - sight_radius)
        y_min = int(y_med - sight_radius)
        x_max = int(x_med + sight_radius)
        y_max = int(y_med + sight_radius)
        fixed_sight = self.frame[y_min:y_max, x_min:x_max]
        height, width = fixed_sight.shape[0:2]
        mask = np.zeros((height, width), np.uint8)
        cv2.circle(mask, (int(sight_radius), int(sight_radius)),
                   int(sight_radius), (255, 255, 255), thickness=-1)
        rotated_sight = cv2.bitwise_and(fixed_sight, fixed_sight, mask=mask)
        rotated_sight = imutils.rotate(rotated_sight, ang)

        if self.save_render:
            cv2.imwrite("%s/sight_%s_%06d.png" %
                        (self.path, veh_id, self.time),
                        rotated_sight)
        if "gray" in self.mode:
            return rotated_sight[:, :, 0]
        else:
            return rotated_sight

    def _add_lane_polys(self):
        """Render road network polygons."""
        for lane_poly, lane_color in zip(self.lane_polys, self.lane_colors):
            self._add_line(lane_poly, lane_color)

    def _add_vehicle_polys(self, orientations, colors, sight_radius):
        """Render vehicle polygons.

        Parameters
        ----------
        orientations : list
            A list of orientations
            An orientation is a list contains [x, y, angle].
        colors : list
            A list of colors corresponding to the vehicle orientations
        sight_radius : int
            Set the radius of observation for RL vehicles (meter)
        """
        for orientation, color in zip(orientations, colors):
            x, y, ang = orientation
            x = (x-self.x_shift)*self.x_scale*self.pxpm
            y = (y-self.y_shift)*self.y_scale*self.pxpm
            self._add_triangle((x, y), ang, 5, color)
            self._add_circle((x, y), sight_radius, color)

    def _add_line(self, lane_poly, lane_color):
        """Render road network polygons.

        Parameters
        ----------
        lane_poly : list
            A list of road network polygons
        lane_color : list
            A list of colors corresponding to the road network polygons
        """
        num = int(len(lane_poly)/2)
        index = [x for x in range(num)]
        group = pyglet.graphics.Group()
        self.lane_batch.add_indexed(
            num, pyglet.gl.GL_LINE_STRIP, group, index,
            ("v2f", lane_poly), ("c4B", lane_color))

    def _add_triangle(self, center, angle, size, color):
        """Render a vehicle as a triangle.

        Parameters
        ----------
        center : tuple
            The center coordinate of the vehicle
        angle : float
            The angle of the vehicle
        size : int
            The size of the rendered triangle
        color : list
            The color of the vehicle  [r, g, b].
        """
        cx, cy = center
        ang = np.radians(angle)
        s = size*self.pxpm
        pt1 = [cx, cy]
        pt1_ = [cx - s*self.x_scale*np.sin(ang),
                cy - s*self.y_scale*np.cos(ang)]
        pt2 = [pt1_[0] + 0.25*s*self.x_scale*np.sin(np.pi/2-ang),
               pt1_[1] - 0.25*s*self.y_scale*np.cos(np.pi/2-ang)]
        pt3 = [pt1_[0] - 0.25*s*self.x_scale*np.sin(np.pi/2-ang),
               pt1_[1] + 0.25*s*self.y_scale*np.cos(np.pi/2-ang)]
        vertex_list = []
        vertex_color = []
        for point in [pt1, pt2, pt3]:
            vertex_list += point
            vertex_color += color
        index = [x for x in range(3)]
        group = pyglet.graphics.Group()
        self.vehicle_batch.add_indexed(
            3, pyglet.gl.GL_POLYGON, group, index,
            ("v2f", vertex_list), ("c4B", vertex_color))

    def _add_circle(self, center, radius, color):
        """Render a vehicle as a circle or render its observation radius.

        Parameters
        ----------
        center : tuple
            The center coordinate of the vehicle
        radius : float
            The size of the rendered vehicle or the radius of observation
        color : list
            The color of the vehicle  [r, g, b].
        """
        if radius == 0:
            return
        cx, cy = center
        radius = radius * self.pxpm
        pxpm = int(self.pxpm*50)
        vertex_list = []
        vertex_color = []
        for idx in range(pxpm):
            angle = np.radians(float(idx)/pxpm * 360.0)
            x = radius*self.x_scale*np.cos(angle) + cx
            y = radius*self.y_scale*np.sin(angle) + cy
            vertex_list += [x, y]
            vertex_color += color
        index = [x for x in range(pxpm)]
        group = pyglet.graphics.Group()
        self.vehicle_batch.add_indexed(
            pxpm, pyglet.gl.GL_LINE_LOOP, group, index,
            ("v2f", vertex_list), ("c4B", vertex_color))

    @staticmethod
    def _truncate_colormap(cmap, minval=0.25, maxval=0.75, n=100):
        """Truncate a matplotlib colormap.

        Parameters
        ----------
        cmap : matplotlib.colors.LinearSegmentedColormap
            Original colormap
        minval : float
            Minimum value of the truncated colormap
        maxval : float
            Maximum value of the truncated colormap
        n : int
            Number of RGB quantization levels of the truncated colormap

        Returns
        -------
        matplotlib.colors.LinearSegmentedColormap
            truncated colormap
        """
        new_cmap = colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'
            .format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap
