import numpy as np
import pyglet
from pyglet.gl import *

window = pyglet.window.Window(256*2, 256*2)

img = np.random.rand(64*64)*255
imgData = (GLubyte * img.size)(*img.astype('uint8'))
pimg = pyglet.image.ImageData(64, 64, 'L', imgData)

def update(dt):
    global pimg
    img = np.random.rand(64*64)*255
    imgData = (GLubyte * img.size)(*img.astype('uint8'))
    pimg = pyglet.image.ImageData(64, 64, 'L', imgData)

pyglet.clock.schedule_interval(update, 1/6.0)

@window.event
def on_draw():
    window.clear()
    pimg.width = 512
    pimg.height = 512
    pimg.blit(0, 0)


pyglet.app.run();