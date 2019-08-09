import numpy as np
import pyglet
import SLM
import PCAM as pc
import time
import MathUtils as mu

import scipy.io as sp

import matplotlib.pyplot as plt

threshold = 6

fig=plt.figure(figsize=(2, 2))

cam = pc.PCAM()

slm = SLM.SLM(3)
slm2 = SLM.SLM(1)

slm2.load_calibration("H1_cal.mat")

slm.enable_blazed()
slm2.enable_blazed()

slm2.set_k_vector(0, 0)
slm2.disable_filter()

slm.set_zernike_coeffs([0, 0, 0, -0.025, -0.5, 0, -.04, 0.1055, 0.1055, -0.04, 0, 0, 0], [0.75])

x = np.linspace(-1, 1, 128*8)
y = np.linspace(-1, 1, 128*8)
xx, yy = np.meshgrid(x, y)

dist = (xx**2) + (yy**2)
img = dist < 1.0
min_size = min(slm.screen_height, slm.screen_width)
slm.set_location_center(slm.screen_width/2, slm.screen_height/2, min_size, min_size)
slm.set_array(img)
slm.enable_filter()

img = np.ones((64, 64)) * np.exp(1.0j*np.pi*0.7)#*0.19)#
slm2.set_array(img)
slm2.set_location_center(slm2.screen_width/2, slm2.screen_height/2, slm2.screen_width, slm2.screen_height)

cam.start()

def update(dt):
    print(pyglet.clock.get_fps())
    cam.fetch_buffer()
    Ih_2d, Iv_2d, Id_2d, Ia_2d = cam.get_pol()
    Ih_2d = np.maximum(Ih_2d - threshold, 0)
    Iv_2d = np.maximum(Iv_2d - threshold, 0)
    Id_2d = np.maximum(Id_2d - threshold, 0)
    Ia_2d = np.maximum(Ia_2d - threshold, 0)
    w = np.size(Ih_2d, 0)
    h = np.size(Ih_2d, 1)
    num = float(w*h)
    print("H: %f V: %f D: %f A: %f" % (np.sum(Ih_2d)/num, np.sum(Iv_2d)/num, np.sum(Id_2d)/num, np.sum(Ia_2d)/num))
    print("H/V: %f D/A: %f" % (np.sum(Ih_2d)/np.sum(Iv_2d), np.sum(Id_2d)/np.sum(Ia_2d)))

    # fig.add_subplot(2, 2, 1)
    # plt.imshow(Ih_2d)

    # fig.add_subplot(2, 2, 2)
    # plt.imshow(Iv_2d)

    # fig.add_subplot(2, 2, 3)
    # plt.imshow(Id_2d)

    # fig.add_subplot(2, 2, 4)
    # plt.imshow(Ia_2d)

    cam.queue_buffer()

    # plt.pause(0.05)

pyglet.clock.schedule_interval(update, 1.0/10.0)

event_loop = pyglet.app.EventLoop()

pyglet.clock.set_fps_limit(10)

@event_loop.event
def on_window_close(window):
    event_loop.exit()

pyglet.app.run()

cam.__del__()

#plt.show()