from OpenGL.GL import *
from OpenGL.GLU import *
import glfw

import time
import cv2
import numpy as np
import traci.constants as tc


# WARNING: This does NOT work yet.
class GLFWRenderer():

    def __init__(self, kernel, save_frame=False, save_dir=None):
        self.kernel = kernel
        self.frame = None
        self.save_frame = save_frame
        self.dpm = 3 # Dots per meter
        if self.save_frame:
            self.save_dir = save_dir

        lower_left, upper_right = self.kernel.simulation.getNetBoundary()
        self.width = int(upper_right[0] - lower_left[0])*self.dpm
        self.height = int(upper_right[1] - lower_left[1])*self.dpm
        # Initialize the library
        if not glfw.init():
            return
        # Set window hint NOT visible
        glfw.window_hint(glfw.VISIBLE, True)
        # Create a windowed mode window and its OpenGL context
        self.window = glfw.create_window(self.width, self.height,
                                         "Window", None, None)
        if not self.window:
            glfw.terminate()
            return
        # Make the window's context current
        glfw.make_context_current(self.window)
        """
        gluPerspective(90, (self.width / self.height), 0.01, 12)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        """
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glBegin(GL_QUADS)
        glColor3f(1, 0, 0)
        glVertex3f(2, 2, 0)
        glVertex3f(2, 2, 2)
        glVertex3f(2, 6, 2)
        glVertex3f(2, 6, 0)
        glEnd()
        time.sleep(10000)
        self.lane_polys = []
        for lane_id in self.kernel.lane.getIDList():
            _lane_poly = self.kernel.lane.getShape(lane_id)
            lane_poly = [i*self.dpm for pt in _lane_poly for i in pt ]
            self.lane_polys += lane_poly

        polys_x = np.asarray(self.lane_polys[::2])
        shift = polys_x.min() - self.dpm*2
        scale = (self.width - self.dpm*4) / (polys_x.max() - polys_x.min())
        self.lane_polys[::2] = [(x-shift)*scale for x in polys_x]
        self.x_shift = shift
        self.x_scale = scale

        polys_y = np.asarray(self.lane_polys[1::2])
        shift = polys_y.min() - self.dpm*2
        scale = (self.height - self.dpm*4) / (polys_y.max() - polys_y.min())
        self.lane_polys[1::2] = [(y-shift)*scale for y in polys_y]
        self.y_shift = shift
        self.y_scale = scale


    def render(self):
        try:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            self.add_lane_polys()
            self.add_vehicle_polys()
            #print(self.kernel.simulation.getCurrentTime())

            buffer = glReadPixels(0, 0, self.width, self.height,
                                  GL_RGB, GL_UNSIGNED_BYTE)
            self.frame = np.frombuffer(buffer,
                                       dtype=np.uint8).reshape(self.width,
                                                               self.height,
                                                               3)
            if self.save_frame:
                t = self.kernel.simulation.getCurrentTime()
                cv2.imwrite("%s/frame%06d.png" % (self.save_dir, t), self.frame)
        except:
            self.close()
        print(self.frame.mean())
        return self.frame


    def close(self):
        glfw.destroy_window(self.window)
        glfw.terminate()


    def add_lane_polys(self):
        for lane_id in self.kernel.lane.getIDList():
            _polys = self.kernel.lane.getShape(lane_id)
            polys = [i for pt in _polys for i in pt]
            polys[::2] = [(x*self.dpm-self.x_shift)*self.x_scale
                          for x in polys[::2]]
            polys[1::2] = [(y*self.dpm-self.y_shift)*self.y_scale
                           for y in polys[1::2]]
            colors = [c for _ in _polys for c in [255, 255, 0]]
            self._add_lane_poly(polys, colors)


    def _add_lane_poly(self, polys, colors):
        num = int(len(polys)/2)
        for idx in range(num):
            c = colors[3*idx:3*idx+3]
            p = polys[2*idx:2*idx+2]
            glBegin(GL_LINE_STRIP)
            glColor3i(c[0], c[1], c[2])
            for v1, v2 in zip(p[::2], p[1::2]):
                glVertex2f(v1, v2)
            glEnd()

    def add_vehicle_polys(self):
        for veh_id in self.kernel.vehicle.getIDList():
            x, y = self.kernel.vehicle.getPosition(veh_id)
            x = (x*self.dpm-self.x_shift)*self.x_scale
            y = (y*self.dpm-self.y_shift)*self.y_scale
            ang = self.kernel.vehicle.getAngle(veh_id)
            c = self.kernel.vehicle.getColor(veh_id)
            sc = self.kernel.vehicle.getShapeClass(veh_id)
            self._add_vehicle_poly_triangle((x, y), ang, 4.5, c)
            #self._add_vehicle_poly_circle((x, y), 3, c)


    def _add_vehicle_poly_triangle(self, center, angle, size, color):
        cx, cy = center
        #print(angle)
        ang = np.radians(angle)
        s = size*self.dpm
        pt1 = [cx, cy]
        pt1_ = [cx - s*np.sin(ang),
                cy - s*np.cos(ang)]
        pt2 = [pt1_[0] + 0.25*s*np.sin(np.pi/2-ang),
               pt1_[1] - 0.25*s*np.cos(np.pi/2-ang)]
        pt3 = [pt1_[0] - 0.25*s*np.sin(np.pi/2-ang),
               pt1_[1] + 0.25*s*np.cos(np.pi/2-ang)]
        vertex_list = []
        for point in [pt1, pt2, pt3]:
            vertex_list += point
        glBegin(GL_POLYGON)
        for idx in range(3):
            v = vertex_list[2*idx:2*idx+2]
            glColor3i(255, 0, 0)
            glVertex2f(v[0], v[1])
        glEnd()


    def _add_vehicle_poly_circle(self, center, radius, color):
        cx, cy = center
        r = radius
        dpm = self.dpm*4
        vertex_list = []
        for idx in range(dpm):
            angle = np.radians(float(idx)/dpm * 360.0)
            x = radius*np.cos(angle) + cx
            y = radius*np.sin(angle) + cy
            vertex_list += [x, y]
        glBegin(GL_POLYGON)
        for idx in range(dpm):
            v = vertex_list[2*idx:2*idx+2]
            glColor3i(255, 0, 0)
            glVertex2f(v[0], v[1])
        glEnd()
