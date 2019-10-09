import numpy as np

import glfw

from PIL import Image, ImageFilter

import scipy.io as sp

import PCAM2 as PC
import SLM2 as SLM

import AutoUtils as au
import MathUtils as mu

def directMeasurement(cam, slm1, slm2, Nx, Ny, activation_ratio, phase_rot_, phase_height, circular, img_file_name):
    Num_of_meas = Nx*Ny

    masks = np.zeros((Num_of_meas, Nx*Ny))
    b_vector = np.zeros(Num_of_meas, dtype=complex)

    phase_rot = phase_rot_*np.pi

    #generate phase and amplitude object for slm1
    im = Image.open(img_file_name + '.png')
    np_im = np.array(im)

    if len(np.shape(np_im)) == 3:
        phase_image = phase_height*np_im[:, :, 1]/255.0
    else:
        phase_image = phase_height*np_im[:, :]/255.0

    pimg = np.exp(1.0j*phase_image)

    x = np.linspace(-1, 1, im.width)
    y = np.linspace(-1, 1, im.height)
    xx, yy = np.meshgrid(x, y)

    dist = np.sqrt((xx**2) + (yy**2))
    imgx = abs(xx) < 1
    imgy = abs(yy) < 1
    img = imgx*imgy#phase_image > 0.01#

    if circular:
        img = dist < 0.95

    #find image center on mask slm and zero state phase
    x, y, phase_low = au.automatic_slm_center(cam, slm1, slm2, slm1.screen_height, 9.5/20, 0.5*np.pi)

    slm2_size = 1.25*slm1.screen_height*9.5/20

    slm1.set_array(0.5*img.astype(complex)*pimg)    #set object slm to display our object

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

    for n in range(Num_of_meas):
        print(n)

        mask = np.random.choice([0.0, 1.0], size = (Nx, Ny), p=[(1.0 - activation_ratio), activation_ratio])

        mask_1d = np.reshape(mask, (1, Nx*Ny))

        masks[n, :] = mask_1d

        a, b = au.measure(cam, slm2, phase_low, phase_rot, mask)

        b_vector[n] = b

    sp.savemat('D:/Alejandro/results/sweeps/' + img_file_name + '/test/ar-' + str(activation_ratio) + 'pr-' + str(phase_rot_) + '.mat', {
        'b': b_vector,
        'masks': masks,
        'Nx': Nx,
        'Ny': Ny,
        'alpha': phase_rot,
        'activation': activation_ratio
        })
