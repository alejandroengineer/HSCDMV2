import numpy as np
import SLM2
import glfw
import PCAM2
import AutoUtils as au
import MathUtils as mu

cam = PCAM2.PCAM()
cam.start()

phase1 = 0.21
phase2 = 1.19

SLM2.init()

slm = SLM2.SLM(3)
slm2 = SLM2.SLM(2)

slm.enable_blazed()

slm.set_k_vector(2.0*np.pi/17.0, 2.0*np.pi/11.0)

slm2.enable_blazed()

slm2.set_k_vector(0, 0)
slm2.disable_filter()

slm2.set_zero(np.exp(1.0j*np.pi*phase1))

slm2.load_calibration('D:\Alejandro\slm cals\H4_cal.mat')

x = np.linspace(-1, 1, 128*4)
y = np.linspace(-1, 1, 128*4)
xx, yy = np.meshgrid(x, y)

dist = np.sqrt((xx**2) + (yy**2))
img = dist < 1

slm.set_array(img.astype(float))

min_size = min(slm.screen_height, slm.screen_width)
slm.set_location_center(slm.screen_width/2, slm.screen_height/2, min_size, min_size)

slm.draw()
slm.swap_buffers()

slm2.set_location(0, 0, 0, 0)

slm2.draw()
slm2.swap_buffers()

au.automatic_exposure_and_framing(cam, 600, 3400*16, 200*16)

while not glfw.window_should_close(slm.window):
    cam.fetch_buffer()
    H, V, D, A = cam.get_pol()
    center_x, center_y = mu.center_cam(V)
    cam.queue_buffer()
    H_sum = mu.circular_integral_fast(H, center_x, center_y, 10)
    V_sum = mu.circular_integral_fast(V, center_x, center_y, 10)
    D_sum = mu.circular_integral_fast(D, center_x, center_y, 10)
    A_sum = mu.circular_integral_fast(A, center_x, center_y, 10)

    print((H_sum/V_sum, D_sum/A_sum, H_sum/D_sum))
    glfw.poll_events()

cam.stop()

glfw.terminate()
