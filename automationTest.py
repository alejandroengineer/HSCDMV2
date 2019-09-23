import PCAM2
import SLM2
import AutoUtils as au
import matplotlib.pyplot as plt
import numpy as np

phase1 = 0.16
phase2 = 1.19

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

x, y, tmp, b_img, tmp2 = au.automatic_slm_center(cam, slm, slm2, slm.screen_height, 9.5/20, np.pi*phase2)

fig=plt.figure(figsize=(2, 2))

cam.fetch_buffer()
H, V, D, A = cam.get_pol()

fig.add_subplot(2, 2, 1)
plt.imshow(np.abs(b_img))

fig.add_subplot(2, 2, 2)
plt.imshow(np.abs(tmp))

fig.add_subplot(2, 2, 3)
plt.imshow(np.abs(tmp2))

fig.add_subplot(2, 2, 4)
plt.imshow(A)

cam.queue_buffer()

cam.__del__()

plt.show()