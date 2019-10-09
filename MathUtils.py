from scipy import ndimage, misc

import numpy as np
import math
import sys

def LG(nx, ny, waist = 20, mode = 1):
    out = np.zeros((nx, ny))
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    xx, yy = np.meshgrid(x, y)

    dist = (xx**2) + (yy**2)

    mag = np.sqrt(dist) * np.exp(-dist*waist)

    max_mag = np.amax(mag)

    mag = mag/max_mag

    arg = np.arctan2(yy, xx)

    return mag*np.exp(mode*1.0j*arg)

def zernike_index(index):
    n = np.floor(np.sqrt(2*(index))-1)
    m = 2*index - n*(n+2)

    if np.abs(m) >  n:
        n = n+1
        m = 2*index - n*(n+2)

    return int(n), int(m)

def zernike(nx, ny, index, mode = 'center', mask = False):
    n, m = zernike_index(index)

    z = np.zeros((ny, nx))
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    if mode == 'center':
        if nx > ny:
            x = x*float(nx)/float(ny)
        else:
            y = y*float(ny)/float(nx)
    elif mode == 'fill':
        if nx < ny:
            x = x*float(nx)/float(ny)
        else:
            y = y*float(ny)/float(nx)
    xx, yy = np.meshgrid(x, y)

    pLength = int((n-abs(m))/2)

    r = np.sqrt((xx**2) + (yy**2))
    a = np.arctan2(yy, xx)

    if(m < 0):
        r2z = np.sin(m*a)
    else:
        r2z = np.cos(m*a)

    R = np.zeros((ny, nx))

    for k in range(pLength+1):
        coeff = (((-1)**k)*math.factorial(n-k)/(math.factorial(k)*math.factorial(((n+m)/2) - k)*math.factorial(((n-m)/2) - k)))
        R = R + (r**(n - 2*k))*coeff
    
    z = r2z*R

    if mask == True:
        mask_ = (r < 1)
        z = z*mask_

    return z

def zernike_profile(nx, ny, coeffs, mode = 'center'):
    n = len(coeffs)

    out = np.zeros((ny, nx))

    for k in range(n):
        if coeffs[k] != 0 and coeffs[k] != 0.0:
            out = out + (coeffs[k]*zernike(nx, ny, k, mode = mode))
    
    return out

def zernike_c_profile(nx, ny, pCoeffs, aCoeffs, mode = 'center'):
    phase = zernike_profile(nx, ny, pCoeffs, mode = mode)
    amp = zernike_profile(nx, ny, aCoeffs, mode = mode)

    out = amp*np.exp(1.0j*np.pi*phase)

    return out

def inverse_sinc(y):    #calculates the inverse of sinc through the use of newtons method
    if y == 1:
        return 0

    x = np.sqrt(6*(1 - y))

    for n in range(100):
        f = (np.sin(x)/x) - y
        df = (np.cos(x)/x) - (np.sin(x)/(x*x))
        x = x - (f/df)

    return x/np.pi

def center_cam(cam, threshold = 0.4):
    cam2 = cam - threshold
    cam2 = cam2*(cam2 > 0)
    Nx = np.size(cam2, 0)
    Ny = np.size(cam2, 1)

    x = np.linspace(0, Nx-1, Nx)
    y = np.linspace(0, Ny-1, Ny)

    yy, xx = np.meshgrid(y, x)

    sum = np.sum(np.sum(cam2))

    x = np.sum(np.sum(xx*cam2))/sum
    y = np.sum(np.sum(yy*cam2))/sum

    ind = np.unravel_index(np.argmax(cam, axis=None), cam.shape)
    x = ind[0]
    y = ind[1]

    return x, y

def circular_integral_fast(input_, cx_, cy_, r):
    cx_f = np.floor(cx_)
    cy_f = np.floor(cy_)
    r_f = np.floor(r) + 3

    Nx = np.size(input_, 0)
    Ny = np.size(input_, 1)

    ex = np.floor(max(0, cx_f - r_f))
    ey = np.floor(max(0, cy_f - r_f))

    input = input_[int(ex):int(min(Nx-1, cx_f + r_f)), int(ey):int(min(Ny-1, cy_f + r_f))]

    return circular_integral(input, cx_ - ex, cy_ - ey, r)

def circular_integral(input, cx, cy, r):
    Nx = np.size(input, 0)
    Ny = np.size(input, 1)

    x = np.linspace(0, Nx-1, Nx)
    y = np.linspace(0, Ny-1, Ny)

    yy, xx = np.meshgrid(y, x)

    dist = ((xx - cx)**2) + ((yy - cy)**2)

    mask = (dist < (r**2))

    norm = np.sum(mask.astype(float))

    return np.sum(np.sum(mask*input/norm))

def data_norm(data):
    num_of_samples = np.size(data)

    mean = np.sum(data)/num_of_samples

    std_dev = np.sqrt(np.sum((data - mean)**2))/num_of_samples

    return mean, std_dev

def solveDM(Ih, Iv, Id, Iu, alpha):
    a1 = Ih - Iv;
    a2 = Id - Iu;
    a3 = Ih+Iv+Id+Iu;

    if 4*a1**2 + 4*a2**2 > (a3)**2:
        print('result is complex!')
        print((Ih, Iv, Id, Iu))
        br = 1.0j*((4*a1**2 + 4*a2**2 - a3**2)/(2*a1 + a3))**(1/2)/2 + (2*a1 + a3)**(1/2)/2
        gr = (2*a1 + a3)**(1/2)/2 - 1.0j*((4*a1**2 + 4*a2**2 - a3**2)/(2*a1 + a3))**(1/2)/2
    else:
        br = (-(4*a1**2 + 4*a2**2 - a3**2)/(2*a1 + a3))**(1/2)/2 + (2*a1 + a3)**(1/2)/2
        gr = (2*a1 + a3)**(1/2)/2 - (-(4*a1**2 + 4*a2**2 - a3**2)/(2*a1 + a3))**(1/2)/2
        
    gi = a2/(2*a1 + a3)**(1/2)

    beta = br - 1.0j*gi
    gamma = gr + 1.0j*gi

    B = gamma/(1 - np.exp(1.0j*alpha))
    A = (beta - (1 + np.exp(1.0j*alpha))*B)/2

    return A, B

def bit_location(i):
    for n in range(32):
        if i >> n == 1:
            return n
    return 32

def swap_even_odd_bits(i):
    i = int(i)
    a = int(715827882)
    b = int(357913941)

    even = i&a
    odd = i&b

    return (even>>1)|(odd<<1)

def bit_reverse(i, N):
    out = i&(~((1<<N) - 1))
    for n in range(N):
        out = out|(((i>>n)&1) << (N - n - 1))
    return out

def stripe_bits(i, slice):
    out = 0
    for n in range(slice):
        out = out|((((i>>n)&1)|((i>>(n+slice-1))&2))<<(n<<1))
    return out

def hadamard_element(i, j, N):
    i = int(i)
    j = int(j)
    out = i&j
    n = 1
    while n < N:
        out = out ^ (out>>n)
        n = n<<1
    return out&1

def hadamard(N):
    out = np.ones((N, N))
    logN = bit_location(N)
    for i in range(N):
        for j in range(N):
            out[i, j] = hadamard_element(i, j, logN)
    return out

def hadamard_masks(size):   #returns a hadamard matrix for square image of width size
    N = size*size
    out = np.ones((N, N))
    logN = bit_location(N)
    for i in range(N):
        i_ = swap_even_odd_bits(bit_reverse(i, logN))
        for j in range(N):
            out[i, j] = hadamard_element(i, j, logN)#stripe_bits(j, logN>>1)
    return out

def diamond(size):
    out = np.ones((int((size+2)/np.sqrt(2)), int((size+2)/np.sqrt(2))))

    out = ndimage.rotate(out, 45, reshape=True)

    return out