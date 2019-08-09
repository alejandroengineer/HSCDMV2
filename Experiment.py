import numpy as np
import pyglet
import SLM
import PCAM as pc
import time
import MathUtils as mu
import matplotlib.pyplot as plt

from PIL import Image, ImageFilter

import scipy.io as sp

Nx = 16
Ny = 16

fig=plt.figure(figsize=(2, 2))

slm = SLM.SLM(3)
slm2 = SLM.SLM(1)

slm.enable_blazed()
slm2.enable_blazed()

slm2.set_k_vector(0, 0)
slm2.disable_filter()

slm2.load_calibration("H1_cal.mat")

fps_ = 60

pmillis = int(round(time.time() * 1000))

num_of_frames = 0

#slm.set_zernike_coeffs([0, 0, 0, -0.3, 0, 0, -0.2, 0.25, 0.25, -0.2, 0, 0, 0], [0.75])
#slm.set_zernike_coeffs([0, 0, 0, -0.17, -0.56, 0, -0.0844, 0.1055, 0.1055, -0.0844, 0, 0, 0], [0.75])
slm.set_zernike_coeffs([0, 0, 0, -0.025, -0.5, 0, -.04, 0.1055, 0.1055, -0.04, 0, 0, 0], [0.75])

x = np.linspace(-1, 1, 128*4)
y = np.linspace(-1, 1, 128*4)
xx, yy = np.meshgrid(x, y)

im = Image.open("abc.png")
im = im.filter(ImageFilter.BLUR)
np_im = np.array(im)

phase_image = 1.5*np.pi*np_im[:, :, 1]/255.0

pimg = np.exp(1.0j*phase_image)

dist = (xx**2) + (yy**2)
img = dist < 1.0
min_size = min(slm.screen_height, slm.screen_width)
slm.set_location_center(slm.screen_width/2, slm.screen_height/2, min_size, min_size)
slm.set_array(img)#*pimg)
slm.enable_filter()

cam = pc.PCAM()

cam.start()

H_list = []
V_list = []
D_list = []
A_list = []

masks = np.empty((0, Nx*Ny))

count = 0

def update(dt):
    global count
    global H_list
    global V_list
    global D_list
    global A_list
    global masks

    print(pyglet.clock.get_fps())
    cam.fetch_buffer()

    H, V, D, A = cam.get_pol()

    cx, cy = mu.center_cam(V, 0.25)

    hi = mu.circular_integral_fast(H, cx, cy, 3)/4096.0
    vi = mu.circular_integral_fast(V, cx, cy, 3)/4096.0
    di = mu.circular_integral_fast(D, cx, cy, 3)/4096.0
    ai = mu.circular_integral_fast(A, cx, cy, 3)/4096.0

    img = np.zeros((Nx, Ny))

    if count > 0:
        H_list.append(hi)
        V_list.append(vi)
        D_list.append(di)
        A_list.append(ai)
    
    if count > 4:
        img = np.random.choice([0, 1], size = (Nx, Ny), p=[0.8, 0.2])

    img_1d = np.reshape(img, (1, Nx*Ny))

    masks = np.append(masks, img_1d, 0)

    slm2.set_array(np.exp(1.0j*(np.pi*0.4 + img*np.pi)))
    slm2.set_location_center(slm2.screen_width/2, slm2.screen_height/2, 512, 512)
    slm2.disable_filter()

    fig.add_subplot(2, 2, 1)
    plt.imshow(H)

    fig.add_subplot(2, 2, 2)
    plt.imshow(V)

    fig.add_subplot(2, 2, 3)
    plt.imshow(D)

    fig.add_subplot(2, 2, 4)
    plt.imshow(A)

    cam.queue_buffer()

    plt.pause(0.01)

    print(count)

    count = count + 1

pyglet.clock.schedule_interval(update, 1/60.0)

event_loop = pyglet.app.EventLoop()

@event_loop.event
def on_window_close(window):
    event_loop.exit()

pyglet.app.run()

sp.savemat('first_run.mat', {
    'H': np.array(H_list),
    'V': np.array(V_list),
    'D': np.array(D_list),
    'U': np.array(A_list),
    'masks': masks
    })

cam.__del__()

plt.show()