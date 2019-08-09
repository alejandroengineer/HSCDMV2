import numpy as np
import pyglet
import SLM
import PCAM as pc
import time
import MathUtils as mu

import scipy.io as sp

slm = SLM.SLM(1)

slm.load_calibration("H1_cal.mat")

slm.enable_blazed()

#slm.set_zernike_coeffs([0, 0, 0, -0.025, -0.5, 3, -.04, 0.1055, 0.1055, -0.04, 0, 0, 0], [0.75])

x = np.linspace(-1, 1, 128*8)
y = np.linspace(-1, 1, 128*8)
xx, yy = np.meshgrid(x, y)

dist = (xx**2) + (yy**2)
img = dist < 1.0
img = np.ones((64, 64)) * np.exp(1.0j*np.pi*0.4)
min_size = min(slm.screen_height, slm.screen_width)
slm.set_location_center(slm.screen_width/2, slm.screen_height/2, slm.screen_width, slm.screen_height)
slm.set_array(img)
slm.set_k_vector(0, 0)
slm.enable_filter()

def update(dt):
    print(pyglet.clock.get_fps())

pyglet.clock.schedule_interval(update, 1.0/60.0)

event_loop = pyglet.app.EventLoop()

pyglet.clock.set_fps_limit(60)

@event_loop.event
def on_window_close(window):
    event_loop.exit()

pyglet.app.run()