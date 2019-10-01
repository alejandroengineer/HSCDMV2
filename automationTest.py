import PCAM2
import SLM2

import AutoUtils as au
import MathUtils as mu

import matplotlib.pyplot as plt
import numpy as np

import glfw

from pyglfw.libapi import *

phase1 = 0.28
phase2 = 0.2#phase1 + 0.5#1.19-0.7

SLM2.init()

slm = SLM2.SLM(3)
slm2 = SLM2.SLM(2)

slm.enable_blazed()
slm2.enable_blazed()

slm.set_k_vector(2.0*np.pi/17.0, 2.0*np.pi/11.0)

slm2.set_k_vector(0, 0)
slm2.disable_filter()

slm2.set_zero(np.exp(1.0j*np.pi*phase1))

slm2.load_calibration("D:\Alejandro\slm cals\H4_cal.mat")

slm.set_zernike_coeffs([0, 2.07, 4, -0.025, 0, 0, -.04, 0.1055, 0.1055, -0.04, 0, 0, 0], [1])

cam = PCAM2.PCAM()
cam.start()

x = np.linspace(-1, 1, 128*4)
y = np.linspace(-1, 1, 128*4)
xx, yy = np.meshgrid(x, y)

dist = np.sqrt((xx**2) + (yy**2))
imgx = abs(xx) < 1
imgy = abs(yy) < 1
img = imgx*imgy
min_size = min(slm.screen_height, slm.screen_width)
slm.set_location_center(slm.screen_width/2, slm.screen_height/2, min_size, min_size)
slm.set_array(0.25*img.astype(float))
slm.enable_filter()

slm.draw()
slm.swap_buffers()

#au.automatic_exposure_and_framing(cam, 400, 3400, 200)

x, y, tmp, b_img, tmp2, phase_low = au.automatic_slm_center(cam, slm, slm2, slm.screen_height, 9.5/20, np.pi*phase2, np.pi*phase1)

fig=plt.figure(figsize=(3, 2))

slm_phase = 0;


while not glfw.window_should_close(slm.window):
    state = glfw.get_mouse_button(slm.window, glfw.MOUSE_BUTTON_LEFT)

    if state == glfw.PRESS:
        slm_phase = 1.0 - slm_phase
        slm2.set_array(np.ones((64, 64))*np.exp(1.0j*(phase_low + 0.5*slm_phase)))
        slm2.draw()
        slm2.swap_buffers()

    cam.fetch_avg()
    H, V, D, A = cam.get_pol()

    fig.add_subplot(3, 2, 1)
    plt.imshow(np.absolute(b_img))

    fig.add_subplot(3, 2, 2)
    plt.imshow(np.absolute(tmp2))

    fig.add_subplot(3, 2, 3)
    plt.imshow(np.angle(b_img))

    fig.add_subplot(3, 2, 4)
    plt.imshow(V)

    fig.add_subplot(3, 2, 5)
    plt.imshow(D)

    fig.add_subplot(3, 2, 6)
    plt.imshow(A)

    center_x, center_y = mu.center_cam(V, np.max(V)*0.7)
    V_value = mu.circular_integral(V, center_x, center_y, 5)
    H_value = mu.circular_integral(H, center_x, center_y, 5)/V_value
    D_value = mu.circular_integral(D, center_x, center_y, 5)/V_value
    A_value = mu.circular_integral(A, center_x, center_y, 5)/V_value
    V_value = 1

    HVtoDA = (H_value+V_value)/(D_value+A_value)

    a2, b2 = mu.solveDM(H_value, V_value, D_value*HVtoDA, A_value*HVtoDA, np.pi*0.5)

    print((HVtoDA, np.absolute(b2)))

    cam.queue_buffer()

    plt.pause(0.05)

    glfw.poll_events()

cam.__del__()

plt.show()