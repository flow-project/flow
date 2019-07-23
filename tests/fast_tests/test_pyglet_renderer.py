from flow.renderer.pyglet_renderer import PygletRenderer as Renderer
import numpy as np
import os
import unittest
import ctypes


class TestPygletRenderer(unittest.TestCase):
    """Tests pyglet_renderer"""

    def setUp(self):
        path = os.path.dirname(os.path.abspath(__file__))[:-11]
        self.data = np.load(
            '{}/data/renderer_data/replay.npy'.format(path),
            allow_pickle=True
        )
        # Default renderer parameters
        self.network = self.data[0]
        self.mode = "drgb"
        self.save_render = False
        self.sight_radius = 25
        self.pxpm = 3
        self.show_radius = True
        self.alpha = 0.9

    def tearDown(self):
        self.renderer.close()

    def test_init(self):
        # Initialize a pyglet renderer
        self.renderer = Renderer(
            self.network,
            mode=self.mode,
            save_render=self.save_render,
            sight_radius=self.sight_radius,
            pxpm=self.pxpm,
            show_radius=self.show_radius,
            alpha=self.alpha
        )

        # Ensure that the attributes match their correct values
        self.assertEqual(self.renderer.mode, self.mode)
        self.assertEqual(self.renderer.save_render, self.save_render)
        self.assertEqual(self.renderer.sight_radius, self.sight_radius)
        self.assertEqual(self.renderer.pxpm, self.pxpm)
        self.assertEqual(self.renderer.show_radius, self.show_radius)
        self.assertEqual(self.renderer.alpha, self.alpha)

    def test_render_drgb(self):
        # Initialize a pyglet renderer
        self.renderer = Renderer(
            self.network,
            mode=self.mode,
            save_render=self.save_render,
            sight_radius=self.sight_radius,
            pxpm=self.pxpm,
            show_radius=self.show_radius,
            alpha=self.alpha
        )

        _human_orientations, _machine_orientations, \
            _human_dynamics, _machine_dynamics, \
            _human_logs, _machine_logs = self.data[100]
        frame = self.renderer.render(
            _human_orientations, _machine_orientations,
            _human_dynamics, _machine_dynamics,
            _human_logs, _machine_logs
        )
        self.assertEqual(self.renderer.mode, 'drgb')
        self.assertEqual(frame.shape, (378, 378, 3))

    def test_render_rgb(self):
        self.mode = 'rgb'
        # Initialize a pyglet renderer
        self.renderer = Renderer(
            self.network,
            mode=self.mode,
            save_render=self.save_render,
            sight_radius=self.sight_radius,
            pxpm=self.pxpm,
            show_radius=self.show_radius,
            alpha=self.alpha
        )

        _human_orientations, _machine_orientations, \
            _human_dynamics, _machine_dynamics, \
            _human_logs, _machine_logs = self.data[100]
        frame = self.renderer.render(
            _human_orientations, _machine_orientations,
            _human_dynamics, _machine_dynamics,
            _human_logs, _machine_logs
        )
        self.assertEqual(self.renderer.mode, 'rgb')
        self.assertEqual(frame.shape, (378, 378, 3))

    def test_render_dgray(self):
        self.mode = 'dgray'
        # Initialize a pyglet renderer
        self.renderer = Renderer(
            self.network,
            mode=self.mode,
            save_render=self.save_render,
            sight_radius=self.sight_radius,
            pxpm=self.pxpm,
            show_radius=self.show_radius,
            alpha=self.alpha
        )

        _human_orientations, _machine_orientations, \
            _human_dynamics, _machine_dynamics, \
            _human_logs, _machine_logs = self.data[100]
        frame = self.renderer.render(
            _human_orientations, _machine_orientations,
            _human_dynamics, _machine_dynamics,
            _human_logs, _machine_logs
        )
        self.assertEqual(self.renderer.mode, 'dgray')
        self.assertEqual(frame.shape, (378, 378))

    def test_render_gray(self):
        self.mode = 'gray'
        # Initialize a pyglet renderer
        self.renderer = Renderer(
            self.network,
            mode=self.mode,
            save_render=self.save_render,
            sight_radius=self.sight_radius,
            pxpm=self.pxpm,
            show_radius=self.show_radius,
            alpha=self.alpha
        )

        _human_orientations, _machine_orientations, \
            _human_dynamics, _machine_dynamics, \
            _human_logs, _machine_logs = self.data[100]
        frame = self.renderer.render(
            _human_orientations, _machine_orientations,
            _human_dynamics, _machine_dynamics,
            _human_logs, _machine_logs
        )
        self.assertEqual(self.renderer.mode, 'gray')
        self.assertEqual(frame.shape, (378, 378))

    def test_get_sight(self):
        # Initialize a pyglet renderer
        self.renderer = Renderer(
            self.network,
            mode=self.mode,
            save_render=self.save_render,
            sight_radius=self.sight_radius,
            pxpm=self.pxpm,
            show_radius=self.show_radius,
            alpha=self.alpha
        )

        _human_orientations, _machine_orientations, \
            _human_dynamics, _machine_dynamics, \
            _human_logs, _machine_logs = self.data[101]

        self.renderer.render(
            _human_orientations, _machine_orientations,
            _human_dynamics, _machine_dynamics,
            _human_logs, _machine_logs
        )
        orientation = self.data[101][0][0]
        id = self.data[101][4][0][-1]
        sight = self.renderer.get_sight(orientation, id)
        self.assertEqual(sight.shape, (150, 150, 3))

    def test_save_renderer(self):
        self.save_render = True
        # Initialize a pyglet renderer
        self.renderer = Renderer(
            self.network,
            mode=self.mode,
            save_render=self.save_render,
            path='/tmp',
            sight_radius=self.sight_radius,
            pxpm=self.pxpm,
            show_radius=self.show_radius,
            alpha=self.alpha
        )
        _human_orientations, _machine_orientations, \
            _human_dynamics, _machine_dynamics, \
            _human_logs, _machine_logs = self.data[101]

        self.renderer.render(
            _human_orientations, _machine_orientations,
            _human_dynamics, _machine_dynamics,
            _human_logs, _machine_logs
        )

        save_path = self.renderer.close()
        saved_data = np.load(save_path, allow_pickle=True)

        self.assertEqual(self.data[0], saved_data[0])
        self.assertEqual(self.data[101], saved_data[1])

    def test_close(self):
        # Initialize a pyglet renderer
        self.renderer = Renderer(
            self.network,
            mode=self.mode,
            save_render=self.save_render,
            sight_radius=self.sight_radius,
            pxpm=self.pxpm,
            show_radius=self.show_radius,
            alpha=self.alpha
        )

        self.renderer.close()

        _human_orientations, _machine_orientations, \
            _human_dynamics, _machine_dynamics, \
            _human_logs, _machine_logs = self.data[1]
        self.assertRaises(
            ctypes.ArgumentError,
            self.renderer.render,
            _human_orientations,
            _machine_orientations,
            _human_dynamics,
            _machine_dynamics,
            _human_logs,
            _machine_logs
        )


if __name__ == '__main__':
    unittest.main()
