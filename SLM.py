import numpy as np
import pyglet
from pyglet.gl import *
from pyglet.gl import gl
from pyshaders import from_files_names, ShaderCompilationError
from OpenGL.GL import *
from OpenGL.GLU import *

def inverse_sinc(y):
    if y == 1:
        return 0

    x = np.sqrt(6*(1 - y))

    for n in range(100):
        f = (np.sin(x)/x) - y
        df = (np.cos(x)/x) - (np.sin(x)/(x*x))
        x = x - (f/df)

    return x/np.pi

def list_screens():
    return pyglet.canvas.get_display().get_screens()

def texture_from_array_RG32F(array):
    width = np.size(array, 0)
    height = np.size(array, 1)
    texture = glGenTextures(1)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, width, height, 0, GL_RG, GL_FLOAT, array)
    return texture

def texture_from_array_1D(array):
    length = np.size(array, 0)
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_1D, texture)
    glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexImage1D(GL_TEXTURE_1D, 0, GL_R32F, length, 0, GL_RED, GL_FLOAT, array)
    return texture

class SLM(pyglet.window.Window):
    def __init__(self, monitor_id):  #monitorID selects the monitor to use as the SLM output
        self.monitor_id = monitor_id
        screen = pyglet.canvas.get_display().get_screens()[monitor_id]
        super(SLM, self).__init__(screen=screen, fullscreen = True)
        self.shader = None
        self.screen_width = screen.width
        self.screen_height = screen.height
        self.set_location(0, 0, self.screen_width, self.screen_height)
        self.set_array(np.zeros((64, 64)))

        invSinc = np.vectorize(inverse_sinc)

        self.lut = 1 - invSinc(np.linspace(0, 1, 255))

        self.GLlut = texture_from_array_1D(self.lut)

    def add_shader(self, vert, frag):
        try:
            self.shader = from_files_names(vert, frag)
            print(self.shader.uniforms)
            self.texLoc = glGetUniformLocation(self.shader.pid, 'tex')
            self.lutLoc = glGetUniformLocation(self.shader.pid, 'lut')
            self.shader.use()
            glUniform1i(self.texLoc, 0);
            glUniform1i(self.lutLoc, 1);
            self.shader.clear()
        except ShaderCompilationError as e:
            print(e.logs)
            self.shader = None
        self.set_array(np.zeros((64, 64)))

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

        if self.shader == None:
            array_1d = np.reshape(array, width*height)*255
            array_data = (GLubyte * array_1d.size)(*array_1d.astype('uint8'))
            glEnable(GL_TEXTURE_2D)
            self.texture = pyglet.image.ImageData(width, height, 'L', array_data).get_texture()
            glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        else:
            rgArray = np.zeros((width, height, 2), 'float')
            rgArray[..., 0] = np.real(array)
            rgArray[..., 1] = np.imag(array)
            self.texture = texture_from_array_RG32F(rgArray)

    def on_draw(self):              #pyglet on draw function, called by pyglet
        self.clear()
        if self.shader == None:
            self.texture.width = self.w
            self.texture.height = self.h
            self.texture.blit(self.x, self.y)
        else:
            self.shader.use()
            
            glEnable(GL_TEXTURE_2D)
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, self.texture);

            glEnable(GL_TEXTURE_1D)
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_1D, self.GLlut);

            x = self.x
            y = self.y
            w = self.w
            h = self.h
            pyglet.graphics.draw_indexed(4, pyglet.gl.GL_TRIANGLES, [0, 1, 2, 0, 2, 3],
            ('v2f', (x, y, x+w, y, x+w, y+h, x, y+h)),
            ('t2f', (0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0)))
            self.shader.clear()
