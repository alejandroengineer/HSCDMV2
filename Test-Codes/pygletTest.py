import numpy as np
import pyglet
from pyglet.gl import *
from pyglet.gl import gl

window = pyglet.window.Window()

img = np.random.rand(64*64)*255
imgData = (GLubyte * img.size)(*img.astype('uint8'))
pimg = pyglet.image.ImageData(64, 64, 'L', imgData)

def update(dt):
    global pimg
    img = np.random.rand(64*64)*255
    imgData = (GLubyte * img.size)(*img.astype('uint8'))
    glEnable(GL_TEXTURE_2D)
    pimg = pyglet.image.ImageData(64, 64, 'L', imgData).get_texture()
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    pimg.scale = 1
    pimg.width = 512
    pimg.height = 512

pyglet.clock.schedule_interval(update, 1/30.0)

@window.event
def on_draw():
    window.clear()
    pimg.blit(0, 0)


pyglet.app.run();