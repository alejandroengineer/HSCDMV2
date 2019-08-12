import numpy as np
import pyglet
import SLM
import PCAM as pc
import time
import MathUtils as mu
import matplotlib.pyplot as plt

from PIL import Image, ImageFilter

import scipy.io as sp

Nx = 16*2
Ny = 16*2

phase_rot = 0.05

activation_ratio = 0.1

Num_of_meas = 32*32

threshold = 6

fig=plt.figure(figsize=(2, 2))

slm = SLM.SLM(3)
slm2 = SLM.SLM(1)

slm.enable_blazed()
slm2.enable_blazed()

slm2.set_k_vector(0, 0)
slm2.disable_filter()

slm2.load_calibration("D:\Alejandro\slm cals\H4_cal.mat")

fps_ = 60

pmillis = int(round(time.time() * 1000))

num_of_frames = 0

#slm.set_zernike_coeffs([0, 0, 0, -0.3, 0, 0, -0.2, 0.25, 0.25, -0.2, 0, 0, 0], [0.75])
#slm.set_zernike_coeffs([0, 0, 0, -0.17, -0.56, 0, -0.0844, 0.1055, 0.1055, -0.0844, 0, 0, 0], [0.75])
slm.set_zernike_coeffs([0, 2.07, 1.5, -0.025, -0.75, 0, -.04, 0.1055, 0.1055, -0.04, 0, 0, 0], [0.75])

x = np.linspace(-1, 1, 128*4)
y = np.linspace(-1, 1, 128*4)
xx, yy = np.meshgrid(x, y)

im = Image.open("F.png")#"logo_radial_ramp_512.png")#"abc.png")
im = im.filter(ImageFilter.BLUR)
np_im = np.array(im)

phase_image = 2*np.pi*np_im[:, :, 1]/255.0

pimg = np.exp(1.0j*phase_image)

dist = (xx**2) + (yy**2)
img = dist < 0.83
min_size = min(slm.screen_height, slm.screen_width)
slm.set_location_center(slm.screen_width/2, slm.screen_height/2, min_size, min_size)
slm.set_array(img.astype(float))#*np_im[:, :, 1]/255.0)#*pimg)
slm.enable_filter()

cam = pc.PCAM()

cam.start()

H_list = []
V_list = []
D_list = []
A_list = []

#img_list = np.zeros((0, cam.w*cam.h))

masks = np.zeros((Num_of_meas, Nx*Ny))

count = 0

def update(dt):
    global count
    global H_list
    global V_list
    global D_list
    global A_list
    global img_list
    global masks

    print(pyglet.clock.get_fps())

    avg_num = 16
    skip_num = 4

    raw_sum = np.zeros((cam.w, cam.h))
    H_sum = np.zeros((cam.w/2, cam.h/2))
    V_sum = np.zeros((cam.w/2, cam.h/2))
    D_sum = np.zeros((cam.w/2, cam.h/2))
    A_sum = np.zeros((cam.w/2, cam.h/2))
    for n in range(avg_num):
        cam.fetch_buffer()
        if n >= skip_num:
            raw_sum = raw_sum + cam.raw_image
            H, V, D, A = cam.get_pol()
            total_power = np.sum(V)
        cam.queue_buffer()

    cam.raw_image = raw_sum/(avg_num - skip_num)

    #img_list = np.append(img_list, np.reshape(cam.raw_image, (1, cam.w*cam.h)), axis=0)

    H, V, D, A = cam.get_pol()

    total_power = np.sum(V)

    cx, cy = mu.center_cam(V, 0.25)

    hi = mu.circular_integral_fast(H, cx - 0.5, cy + 0.5, 8)/total_power#4096.0
    vi = mu.circular_integral_fast(V, cx, cy, 8)/total_power#4096.0
    di = mu.circular_integral_fast(D, cx, cy + 0.5, 8)/total_power#4096.0
    ai = mu.circular_integral_fast(A, cx - 0.5, cy, 8)/total_power#4096.0

    img = np.zeros((Nx, Ny))

    if count > 0:
        H_list.append(hi)
        V_list.append(vi)
        D_list.append(di)
        A_list.append(ai)
    
    if count > 4:
        img = np.random.choice([0.0, 1.0], size = (Nx, Ny), p=[(1.0 - activation_ratio), activation_ratio])

    img_1d = np.reshape(img, (1, Nx*Ny))

    masks[count, :] = img_1d

    slm2.set_array(np.exp(1.0j*(np.pi*0.24 + (img*phase_rot*0.99))))
    slm2.set_location_center(slm2.screen_width/2, 65 + slm2.screen_height/2, 384, 384)
    slm2.disable_filter()

    # fig.add_subplot(2, 2, 1)
    # plt.imshow(H)

    # fig.add_subplot(2, 2, 2)
    # plt.imshow(V)

    # fig.add_subplot(2, 2, 3)
    # plt.imshow(D)

    # fig.add_subplot(2, 2, 4)
    # plt.imshow(A)

    # plt.pause(0.05)

    print(count)

    count = count + 1

    if count >= Num_of_meas:
        slm.close()
        slm2.close()

pyglet.clock.schedule_interval(update, 1/60.0)

event_loop = pyglet.app.EventLoop()

@event_loop.event
def on_window_close(window):
    event_loop.exit()

pyglet.app.run()

sp.savemat('D:/Alejandro/results/data_190812_run26.mat', {
    'H': np.array(H_list),
    'V': np.array(V_list),
    'D': np.array(D_list),
    'U': np.array(A_list),
    'masks': masks,
    #'raw_image': img_list,
    'Nx': Nx,
    'Ny': Ny,
    'alpha': phase_rot,
    'activation': activation_ratio
    })

cam.__del__()

# plt.show()