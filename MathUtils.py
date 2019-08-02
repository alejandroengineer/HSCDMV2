import numpy as np
import math

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

def zernike(nx, ny, index, mode = 'center'):
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

    mask = (r < 1)

    R = np.zeros((ny, nx))

    for k in range(pLength+1):
        coeff = (((-1)**k)*math.factorial(n-k)/(math.factorial(k)*math.factorial(((n+m)/2) - k)*math.factorial(((n-m)/2) - k)))
        R = R + (r**(n - 2*k))*coeff
    
    z = mask*r2z*R

    return z

def zernike_profile(nx, ny, coeffs, mode = 'center'):
    n = len(coeffs)

    out = np.zeros((ny, nx))

    for k in range(n):
        print(coeffs[k])
        out = out + (coeffs[k]*zernike(nx, ny, k, mode = mode))
    
    return out

def zernike_c_profile(nx, ny, pCoeffs, aCoeffs, mode = 'center'):
    phase = zernike_profile(nx, ny, pCoeffs, mode = mode)
    amp = zernike_profile(nx, ny, aCoeffs, mode = mode)

    out = amp*np.exp(1.0j*np.pi*phase)

    return out