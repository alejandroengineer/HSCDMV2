import numpy as np
import pyglet
from pyglet.gl import *
from pyglet.gl import gl
from pyshaders import from_files_names, ShaderCompilationError 

def list_screens():
    return pyglet.canvas.get_display().get_screens()

class SLM(pyglet.window.Window):
    def __init__(self, monitor_id):  #monitorID selects the monitor to use as the SLM output
        self.monitor_id = monitor_id
        screen = pyglet.canvas.get_display().get_screens()[monitor_id]
        super(SLM, self).__init__(screen=screen, fullscreen = True)
        self.screen_width = screen.width
        self.screen_height = screen.height
        self.set_location(0, 0, self.screen_width, self.screen_height)
        self.set_array(np.zeros((64, 64)))
        self.shader = None

    def add_shader(self, vert, frag):
        try:
            self.shader = from_files_names(vert, frag)
        except ShaderCompilationError as e:
            print(e.logs)
            self.shader = None

    def set_location(self, x, y, w, h): #selects, in pixel space, this location and size of the image
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    
    def set_location_center(self, x, y, w, h): #selects, in pixel space, this location and size of the image
        self.x = x - (w/2)
        self.y = y - (h/2)
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
        if self.shader == None:
            self.texture.blit(self.x, self.y)
        else:
            self.shader.use()
            gl.glActiveTexture(GL_TEXTURE0)
            self.shader.clear()
