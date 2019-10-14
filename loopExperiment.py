import numpy as np

import glfw

from PIL import Image, ImageFilter

import scipy.io as sp

import PCAM2 as PC
import SLM2 as SLM

import AutoUtils as au
import MathUtils as mu

import directMeasurement as dm

Nx = 64
Ny = 64

id = '20191014'

# files = ['Star', 'USFGrayScale', 'logo_radial_ramp_512', 'logo_ramps_512', 'zernike_4_2']
# phase_height = [np.pi, 1.5*np.pi, 1.5*np.pi, 1.5*np.pi, 2*np.pi]
# circular = [True, False, False, False, True]

files = ['spiral_phase_OMA1', 'spiral_phase_OMA2', 'spiral_phase_OMA3', 'logo_radial_ramp_512', 'zernike_4_2']
phase_height = [2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi]
circular = [True, True, True, False, True]

phase_rots = [0.1, 0.2, 0.5, 0.75, 1]
act_rats = [0.05, 0.1, 0.25, 0.5, 0.75]

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

for n in range(len(files)):
    for phase_rot in phase_rots:
        for act_rat in act_rats:
            dm.directMeasurement(cam, slm1, slm2, Nx, Ny, act_rat, phase_rot, phase_height[n], circular[n], files[n], id)

cam.stop()

glfw.terminate()
