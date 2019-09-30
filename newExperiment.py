import numpy as np

import glfw

import PCAM2 as PC
import SLM2 as SLM

import AutoUtils as au
import MathUtils as mu

phase1 = 0.21   #zero phase constant
phase2 = 1.19   #pi phase constant

cam = PC.PCAM()  #start camera
cam.start()

SLM.init()         #initialize glfw

slm1 = SLM.SLM(3)  #slm1 is the object slm
slm2 = SLM.SLM(2)  #slm2 is the mask slm

slm1.enable_blazed()#enable blazed grating shader on the object slm

slm1.set_k_vector(2.0*np.pi/17.0, 2.0*np.pi/11.0)   #set blazed diffraction direction

slm2.enable_blazed()#enable blazed grating shader on the mask slm

slm2.set_k_vector(0, 0) #set blazed grating to flat on the mask slm to utilized the slm calibration 
slm2.disable_filter()

slm2.set_zero(np.exp(1.0j*np.pi*phase1))    #set mask slm background image

slm2.load_calibration('D:\Alejandro\slm cals\H4_cal.mat')   #load mask slm calibration

#generate phase and amplitude object for slm1
x = np.linspace(-1, 1, 128*4)
y = np.linspace(-1, 1, 128*4)
xx, yy = np.meshgrid(x, y)

dist = np.sqrt((xx**2) + (yy**2))
img = dist < 1

slm1.set_array(img.astype(float))    #set object slm to display our object

#set object location and size in slm2
min_size = min(slm1.screen_height, slm1.screen_width)
slm1.set_location_center(slm1.screen_width/2, slm1.screen_height/2, min_size, min_size)

slm1.draw()         #display the object we are going to image
slm1.swap_buffers()

#display the zero phase image on slm2
slm2.set_location(0, 0, 0, 0)
slm2.draw()
slm2.swap_buffers()

#perform automatic exposure and framing of the DC spot
au.automatic_exposure_and_framing(cam, 600, 3400*16, 200*16)


while not glfw.window_should_close(slm1.window):
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
