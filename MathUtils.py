import numpy as np

def LG(nx, ny):
    out = np.zeros((nx, ny))
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    xx, yy = np.meshgrid(x, y)

    dist = (xx**2) + (yy**2)

    mag = np.sqrt(dist) * np.exp(-dist*20.0)

    max_mag = np.amax(mag)

    mag = mag/max_mag

    arg = np.arctan2(yy, xx)

    return mag*np.exp(4.0j*arg)