import numpy as np
import pyglet
import SLM
import PCAM2 as pc
import time
import MathUtils as mu
import matplotlib.pyplot as plt

from PIL import Image, ImageFilter

import scipy.io as sp

show_cam = False

save_datacube = False

raster_scan = False

start_frame = 4

Nx = 16*2
Ny = 16*2

phase1 = 0.16
phase2 = 1.19

phase_rot = np.pi

activation_ratio = 0.1

target_exposure = 3000*16;
exposure_tolerance = 400*16;

Num_of_meas = int(Nx*Ny) + start_frame

threshold = 4

if show_cam:
    fig=plt.figure(figsize=(2, 2))

slm = SLM.SLM(3)
slm2 = SLM.SLM(1)

slm.enable_blazed()
slm2.enable_blazed()

slm.set_k_vector(2.0*np.pi/17.0, 2.0*np.pi/11.0)

slm2.set_k_vector(0, 0)
slm2.disable_filter()

slm2.set_zero(np.exp(1.0j*np.pi*phase1))

slm2.load_calibration("D:\Alejandro\slm cals\H4_cal.mat")

fps_ = 60

pmillis = int(round(time.time() * 1000))

num_of_frames = 0

#slm.set_zernike_coeffs([0, 0, 0, -0.3, 0, 0, -0.2, 0.25, 0.25, -0.2, 0, 0, 0], [0.75])
#slm.set_zernike_coeffs([0, 0, 0, -0.17, -0.56, 0, -0.0844, 0.1055, 0.1055, -0.0844, 0, 0, 0], [0.75])
slm.set_zernike_coeffs([0, 2.07, 4, -0.025, 0, 0, -.04, 0.1055, 0.1055, -0.04, 0, 0, 0], [1])
#


x = np.linspace(-1, 1, 128*4)
y = np.linspace(-1, 1, 128*4)
xx, yy = np.meshgrid(x, y)

im = Image.open("USFGrayScale.png")#logo_ramps_512.png")#"Star.png")#"logo_radial_ramp_512.png")#"abc.png")
#im = im.filter(ImageFilter.BLUR)
np_im = np.array(im)

phase_image = 2.0*np.pi*np_im[:, :, 1]/255.0

pimg = np.exp(1.0j*phase_image)

dist = np.sqrt((xx**2) + (yy**2))
imgx = abs(xx) < 0.6#
imgy = abs(yy) < 0.6
#img = dist < 0.95
img = phase_image > 0.01#imgx*imgy
min_size = min(slm.screen_height, slm.screen_width)
slm.set_location_center(slm.screen_width/2, slm.screen_height/2, min_size, min_size)
slm.set_array(img.astype(float)*pimg)#np_im[:, :, 1]/255.0)#
slm.enable_filter()

cam = pc.PCAM()

cam.start()

H_list = []
V_list = []
D_list = []
A_list = []

cw = int(cam.h/2)
ch = int(cam.w/2)

if save_datacube:
    H_img_list = np.zeros((cw, ch, Num_of_meas))
    V_img_list = np.zeros((cw, ch, Num_of_meas))
    D_img_list = np.zeros((cw, ch, Num_of_meas))
    A_img_list = np.zeros((cw, ch, Num_of_meas))
else:
    H_img_list = np.zeros((cw, ch, 1))
    V_img_list = np.zeros((cw, ch, 1))
    D_img_list = np.zeros((cw, ch, 1))
    A_img_list = np.zeros((cw, ch, 1))

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

    if count == 2:
        good_exposure = False
        while good_exposure == False:
            cam.fetch_buffer()
            max_pixel = np.max(cam.raw_image)
            if abs(max_pixel - target_exposure) > exposure_tolerance:
                new_exposure = target_exposure*cam.exposure/max_pixel
                print((max_pixel, new_exposure))
                if new_exposure <= 10:
                    break
                if new_exposure >= 12078:
                    break
                cam.set_exposure(int(new_exposure))
            else:
                good_exposure = True
            cam.queue_buffer()
            time.sleep(0.2)
        cam.fetch_buffer()
        max_pixel = np.max(cam.raw_image)
        print(max_pixel)
        cam.queue_buffer()


    avg_num = 20
    skip_num = 6

    raw_sum = np.zeros((cam.h, cam.w))
    cw = int(cam.h/2)
    ch = int(cam.w/2)
    H_sum = np.zeros((cw, ch))
    V_sum = np.zeros((cw, ch))
    D_sum = np.zeros((cw, ch))
    A_sum = np.zeros((cw, ch))
    for n in range(avg_num):
        cam.fetch_buffer()
        if n >= skip_num:
            raw_sum = raw_sum + cam.raw_image
            H, V, D, A = cam.get_pol()
            total_power = np.sum(V)
            H_sum = H_sum + (H/total_power)
            V_sum = V_sum + (V/total_power)
            D_sum = D_sum + (D/total_power)
            A_sum = A_sum + (A/total_power)
        cam.queue_buffer()

    H = H_sum
    V = V_sum
    D = D_sum
    A = A_sum

    cam.raw_image = raw_sum/(avg_num - skip_num)

    print(np.max(cam.raw_image))

    if save_datacube:
        H_img_list[:, :, count] = H
        V_img_list[:, :, count] = V
        D_img_list[:, :, count] = D
        A_img_list[:, :, count] = A

    total_power = np.sum(V)

    cx, cy = mu.center_cam(V, np.max(V)*0.7)

    # hi = mu.circular_integral_fast(H, cx - 0.5, cy + 0.5, 3)/total_power#4096.0
    # vi = mu.circular_integral_fast(V, cx, cy, 3)/total_power#4096.0
    # di = mu.circular_integral_fast(D, cx, cy + 0.5, 3)/total_power#4096.0
    # ai = mu.circular_integral_fast(A, cx - 0.5, cy, 3)/total_power#4096.0

    hi = mu.circular_integral_fast(H, cx, cy, 3)/total_power#4096.0
    vi = mu.circular_integral_fast(V, cx, cy, 3)/total_power#4096.0
    di = mu.circular_integral_fast(D, cx, cy, 3)/total_power#4096.0
    ai = mu.circular_integral_fast(A, cx, cy, 3)/total_power#4096.0

    print((cx, cy))

    img = np.zeros((Nx, Ny))

    if count > 0:
        H_list.append(hi)
        V_list.append(vi)
        D_list.append(di)
        A_list.append(ai)
    
    if count >= start_frame:
        if raster_scan:
            img_1d = np.zeros(Nx * Ny)
            img_1d[count - start_frame] = 1.0
            img = np.reshape(img_1d, (Nx, Ny))
        else:
            img = np.random.choice([0.0, 1.0], size = (Nx, Ny), p=[(1.0 - activation_ratio), activation_ratio])

    img_1d = np.reshape(img, (1, Nx*Ny))

    masks[count, :] = img_1d

    slm2.set_array(np.exp(1.0j*(np.pi*phase1 + (img*phase_rot*(phase2 - phase1)))))
    #slm2.set_location_center(slm2.screen_width/2, 65 + slm2.screen_height/2, 384, 384)
    slm2.set_location_center(20 + (slm2.screen_width/2), 80 + (slm2.screen_height/2), 300, 300)
    slm2.disable_filter()

    if show_cam:
        fig.add_subplot(2, 2, 1)
        plt.imshow(H)

        fig.add_subplot(2, 2, 2)
        plt.imshow(V)

        fig.add_subplot(2, 2, 3)
        plt.imshow(D)

        fig.add_subplot(2, 2, 4)
        plt.imshow(A)

        plt.pause(0.05)

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

sp.savemat('D:/Alejandro/results/data_190821_run3.mat', {
    'H': np.array(H_list),
    'V': np.array(V_list),
    'D': np.array(D_list),
    'U': np.array(A_list),
    'masks': masks,
    'H_img_list': H_img_list,
    'V_img_list': V_img_list,
    'D_img_list': D_img_list,
    'A_img_list': A_img_list,
    'Nx': Nx,
    'Ny': Ny,
    'start_frame' : start_frame,
    'alpha': phase_rot,
    'activation': activation_ratio,
    'contains_datacube': save_datacube
    })

cam.__del__()

# plt.show()