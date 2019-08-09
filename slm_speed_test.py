import numpy as np
import pyglet
import SLM
import PCAM as pc
import time
import MathUtils as mu

import scipy.io as sp

cam = pc.PCAM()

cam.start()

slm = SLM.SLM(1)

#slm.enable_blazed()

slm.set_zernike_coeffs([0, 0, 0, -0.025, -0.5, 3, -.04, 0.1055, 0.1055, -0.04, 0, 0, 0], [0.75])

x = np.linspace(-1, 1, 128*8)
y = np.linspace(-1, 1, 128*8)
xx, yy = np.meshgrid(x, y)

dist = (xx**2) + (yy**2)
img = dist < 1.0
min_size = min(slm.screen_height, slm.screen_width)
slm.set_location_center(slm.screen_width/2, slm.screen_height/2, min_size, min_size)
slm.set_array(img)
slm.enable_filter()

count = 0
count2 = 0

mod_list = []
sum_list = []
circle_list = []
tim_list = []
cx_list = []
cy_list = []

img_list = np.zeros((200, 200, 0))

def update(dt):
    global count
    global count2
    global mod_list
    global sum_list
    global tim_list
    global img_list
    global cam

    print(pyglet.clock.get_fps())

    mod = count%6
    if mod == 0:
        count2 = count2 + 1

    mod2 = count2%2

    cam.fetch_buffer()

    img = cam.get_pol()

    mod_list.append(mod2)
    cx, cy = mu.center_cam(cam.raw_image, 0.25)
    circle_list.append(mu.circular_integral_fast(cam.raw_image, cx, cy, 10))
    tim_list.append(time.time())
    sum_list.append(np.sum(np.sum(cam.raw_image)))
    cx_list.append(cx)
    cy_list.append(cy)

    img_list = np.dstack((img_list, cam.raw_image))

    print(mod2)

    #slm.set_k_vector(2.0*np.pi*float(mod2)/13.0, 2.0*np.pi*float(mod2)/11.0)

    count = count + 1

    cam.queue_buffer()

pyglet.clock.schedule_interval(update, 1.0/60.0)

event_loop = pyglet.app.EventLoop()

pyglet.clock.set_fps_limit(60)

@event_loop.event
def on_window_close(window):
    event_loop.exit()

pyglet.app.run()

cam.__del__()

sp.savemat('speed_test13.mat', {
    'mode_flag': np.array(mod_list),
    'power_sum': np.array(sum_list),
    'circle_sum': np.array(circle_list),
    'time_stamp': np.array(tim_list),
    'cx' : np.array(cx_list),
    'cy' : np.array(cy_list),
    'img_list' : img_list
    })