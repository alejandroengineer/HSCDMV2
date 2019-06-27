import numpy as np
import pyglet
from pyglet.gl import *

class SLM(pyglet.window.Window):
    def __init__(self, monitorID):  #monitorID selects the monitor to use as the SLM output
        self.monitorID = monitorID
        screen = pyglet.canvas.get_display().get_screens()[monitorID]
        super(SLM, self).__init__(screen=screen, fullscreen = True)
        self.screen_width = screen.width
        self.screen_height = screen.height
        self.set_location(0, 0, self.screen_width, self.screen_height)

    def set_location(self, x, y, w, h): #selects, in pixel space, this location and size of the image
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    
    def set_array(self, array):     #sets the array which will be drawn (numpy array passed as a 2d array)
                                    #valid array values are from 0 to 1
        width = np.size(array, 0)
        height = np.size(array, 1)
        array_1d = np.reshape(array, width*height)*255
        array_data = (GLubyte * array_1d.size)(*array_1d.astype('uint8'))
        glEnable(GL_TEXTURE_2D)
        self.texture = pyglet.image.ImageData(width, height, 'L', array_data).get_texture()
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

    def on_draw(self):              #pyglet on draw function, called by pyglet
        self.clear()
        self.texture.width = self.w
        self.texture.height = self.h
        self.texture.blit(self.x, self.y)

slm = SLM(1)
slm2 = SLM(0)

def update(dt):
    img = np.random.rand(64, 64)
    slm.set_location(0, slm.screen_height - 512, 512, 512)
    slm.set_array(img)
    slm2.set_location(0, 0, 512, 512)
    slm2.set_array(img)

pyglet.clock.schedule_interval(update, 1/30.0)

pyglet.app.run();
