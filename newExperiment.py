import numpy as np

import glfw

from PIL import Image, ImageFilter

import scipy.io as sp

import PCAM2 as PC
import SLM2 as SLM

import AutoUtils as au
import MathUtils as mu

Nx = 32
Ny = 32
Num_of_meas = Nx*Ny

activation_ratio = 0.25

phase_rot = 0.5*np.pi  #phase rotation amount

masks = np.zeros((Num_of_meas, Nx*Ny))
b_vector = np.zeros(Num_of_meas, dtype=complex)

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

slm2.load_calibration('D:\Alejandro\slm cals\H4_cal.mat')   #load mask slm calibration

#set zernike correction for object plane slm
slm1.set_zernike_coeffs([0, 2.07, 4, -0.025, 0, 0, -.04, 0.1055, 0.1055, -0.04, 0, 0, 0], [1])

#generate phase and amplitude object for slm1
im = Image.open("zernike_4_2.png")#"zernike_9_5_512.png")#"logo_ramps_512.png")#USFGrayScale.png")#"Star.png")#"logo_radial_ramp_512.png")#"abc.png")
#im = im.filter(ImageFilter.BLUR)
np_im = np.array(im)

phase_image = 2.0*np.pi*np_im[:, :]/255.0

pimg = np.exp(1.0j*phase_image)

x = np.linspace(-1, 1, 128*4)
y = np.linspace(-1, 1, 128*4)
xx, yy = np.meshgrid(x, y)

dist = np.sqrt((xx**2) + (yy**2))
imgx = abs(xx) < 1
imgy = abs(yy) < 1
#img = dist < 0.95
img = imgx*imgy#phase_image > 0.01#

#find image center on mask slm and zero state phase
x, y, b_img, phase_low = au.automatic_slm_center(cam, slm1, slm2, slm1.screen_height, 9.5/20, phase_rot)

slm2_size = 1.25*slm1.screen_height*9.5/20

slm1.set_array(0.5*img.astype(float)*pimg)    #set object slm to display our object

#set object location and size in slm2
min_size = min(slm1.screen_height, slm1.screen_width)
slm1.set_location_center(slm1.screen_width/2, slm1.screen_height/2, min_size, min_size)

slm1.draw()         #display the object we are going to image
slm1.swap_buffers()

#reset slm2 display for autoexposure
slm2.set_array(np.ones((16, 16))*np.exp(1.0j*phase_low))
slm2.draw()
slm2.swap_buffers()

#perform automatic exposure and framing of the DC spot
au.automatic_exposure_and_framing(cam, 600, 3100*16, 200*16)

cam.fetch_avg()

H, V, D, A = cam.get_pol()

center_x, center_y = mu.center_cam(V, np.max(V)*0.7)

for n in range(Num_of_meas):
    print(n)

    mask = np.random.choice([0.0, 1.0], size = (Nx, Ny), p=[(1.0 - activation_ratio), activation_ratio])

    mask_1d = np.reshape(mask, (1, Nx*Ny))

    masks[n, :] = mask_1d

    a, b = au.measure(cam, slm2, phase_low, phase_rot, mask)#, (center_x, center_y))

    b_vector[n] = b

sp.savemat('D:/Alejandro/results/data_191007_run21.mat', {
    'b': b_vector,
    'masks': masks,
    'Nx': Nx,
    'Ny': Ny,
    'alpha': phase_rot,
    'activation': activation_ratio
    })

cam.stop()

glfw.terminate()
