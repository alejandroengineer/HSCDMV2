import numpy as np
import pyglet
import SLM
import PCAM
import time
import MathUtils as mu

import quantities as pq
from instrumental import instrument, list_instruments

num_of_coeffs = 20
first_coeff = 1

dx = 0.05
step = 0.5

slm = SLM.SLM(3)
slm2 = SLM.SLM(2)

slm.enable_blazed()

fps_ = 60

pmillis = int(round(time.time() * 1000))

num_of_frames = 0

inst = list_instruments()

print(inst)

cam = instrument(inst[0])

cam.master_gain = 1
cam.gain_boost = False

#cam.start_live_video(framerate = '60 hertz', exposure_time='0.05 millisecond')

current_coeffs = [0.1338858,  -0.04361271,  0.0405758, -0.54168096, -0.05750595,
 -0.06092065,  0.27694177,  0.06299513, -0.14471439,  0.05366629,  0.09215891,
 -0.34178976,  0.39306562,  0.20678763,  0.08826728,  0.02828799, -0.02992248,
 -0.10104176, -0.07010867, -0.02751629,]
#slm.set_zernike_coeffs([0, 0, 0, -0.3, 0, 0, -0.2, 0.25, 0.25, -0.2, 0, 0, 0], [0.75])
tmp = [0]
tmp.extend(current_coeffs)
slm.set_zernike_coeffs(tmp, [1])

x = np.linspace(-1, 1, 128*8)
y = np.linspace(-1, 1, 128*8)
xx, yy = np.meshgrid(x, y)

dist = (xx**2) + (yy**2)
img = dist < 1.0
min_size = min(slm.screen_height, slm.screen_width)
slm.set_location_center(slm.screen_width/2, slm.screen_height/2, min_size, min_size)
slm.set_array(img)
slm.enable_filter()

counter = 0

gradient = np.zeros(num_of_coeffs)

center_total = 0

def update(dt):
    global counter
    global center_total
    global current_coeffs

    mode = counter%(num_of_coeffs*2 + 1)

    #frame_ready = cam.wait_for_frame()
    #if frame_ready:
    #    img2 = cam.latest_frame()/255
    #else:
    #    print('ahh')
    img2 = cam.grab_image(exposure_time='0.0005 millisecond')/255
    cx, cy = mu.center_cam(img2)
    total = mu.circular_integral(img2, cx, cy, 8)/mu.circular_integral(img2, cx, cy, 256)
    #total = mu.circular_integral_fast(img2, cx, cy, 8)

    #print(total - total2)

    slm2.set_array(img2)

    delta = np.zeros(num_of_coeffs)

    if mode < num_of_coeffs*2:
        delta[int(mode/2)] = dx*(1 - 2*(mode%2))

    if mode > 0 and mode%2 == 1:
        center_total = total

    if mode > 0 and mode%2 == 0:
        working_coeff = int(mode/2)-1
        gradient[working_coeff] = (center_total - total)/(2.0*dx)

    if mode == num_of_coeffs*2:
        current_coeffs = current_coeffs + step*gradient

    pcoeffs = np.zeros(num_of_coeffs + first_coeff)
    pcoeffs[first_coeff:(num_of_coeffs + first_coeff)] = current_coeffs + delta
    
    slm.set_zernike_coeffs(pcoeffs, [1])

    if mode == 0:
        print(total)
        print(pcoeffs)

    counter = counter + 1
    #pyglet.clock.get_fps())


pyglet.clock.schedule_interval(update, 1/60.0)

event_loop = pyglet.app.EventLoop()

@event_loop.event
def on_window_close(window):
    event_loop.exit()

pyglet.app.run()