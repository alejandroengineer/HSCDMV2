import numpy as np
import pyglet
import scipy.io as sp
from pyglet.gl import *
from pyglet.gl import gl
from pyshaders import from_files_names, ShaderCompilationError
from OpenGL.GL import *
from OpenGL.GLU import *

def inverse_sinc(y):    #calculates the inverse of sinc through the use of newtons method
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

def gl_enable_filtering():  #useful function for enableing filtering and formating the bound texture
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

def gl_disable_filtering(): #useful function for disableing filtering of the image. This also formats the currenly bound texture
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

class SLM(pyglet.window.Window):
    def __init__(self, monitor_id):  #monitorID selects the monitor to use as the SLM output
        self.monitor_id = monitor_id
        screen = pyglet.canvas.get_display().get_screens()[monitor_id]
        super(SLM, self).__init__(screen=screen, fullscreen = True)
        self.shader = None
        self.screen_width = screen.width
        self.screen_height = screen.height
        self.set_location(0, 0, self.screen_width, self.screen_height)

        self.data = np.zeros((64, 64))
        glEnable(GL_TEXTURE_2D)
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        gl_enable_filtering()
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, np.size(self.data, 0), np.size(self.data, 1), 0, GL_RED, GL_FLOAT, self.data)

        invSinc = np.vectorize(inverse_sinc)

        self.sinc_lut = 1 - invSinc(np.linspace(0, 1, 255))

        glEnable(GL_TEXTURE_1D)
        self.GL_sinc_lut = glGenTextures(1)
        glBindTexture(GL_TEXTURE_1D, self.GL_sinc_lut)
        gl_enable_filtering()
        glTexImage1D(GL_TEXTURE_1D, 0, GL_R32F, np.size(self.sinc_lut, 0), 0, GL_RED, GL_FLOAT, self.sinc_lut)

        self.calA = np.zeros((64, 64, 4))
        self.calA[:, :, 1] = 0.5/np.pi;
        self.GLcalA = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.GLcalA)
        gl_enable_filtering()
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, np.size(self.calA, 1), np.size(self.calA, 0), 0, GL_RGBA, GL_FLOAT, self.calA)

        self.calB = np.zeros((64, 64, 4))
        self.GLcalB = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.GLcalB)
        gl_enable_filtering()
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, np.size(self.calB, 1), np.size(self.calB, 0), 0, GL_RGBA, GL_FLOAT, self.calB)


    def add_shader(self, vert, frag):
        try:
            self.shader = from_files_names(vert, frag)
            print(self.shader.uniforms)
            self.texLoc = glGetUniformLocation(self.shader.pid, 'tex')
            self.lutLoc = glGetUniformLocation(self.shader.pid, 'Alut')
            self.calALoc = glGetUniformLocation(self.shader.pid, 'calA')
            self.calBLoc = glGetUniformLocation(self.shader.pid, 'calB')
            self.dirLoc = glGetUniformLocation(self.shader.pid, 'dir')
            self.ssLoc = glGetUniformLocation(self.shader.pid, 'screen_size')
            self.shader.use()
            glUniform1i(self.texLoc, 0)
            glUniform1i(self.lutLoc, 1)
            glUniform1i(self.calALoc, 2)
            glUniform1i(self.calBLoc, 3)
            self.dir_vector = (2.0*np.pi/13.0, 2.0*np.pi/19.0)
            glUniform2f(self.dirLoc, self.dir_vector[0], self.dir_vector[1])
            glUniform2f(self.ssLoc, self.screen_width, self.screen_height)
            self.shader.clear()
        except ShaderCompilationError as e:
            print(e.logs)
            self.shader = None

    def load_calibration(self, file_name):  #load a calibration file
        calibration = sp.loadmat(file_name)
        self.calA = calibration['calA']
        self.calB = calibration['calB']

        glBindTexture(GL_TEXTURE_2D, self.GLcalA)
        gl_enable_filtering()
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, np.size(self.calA, 1), np.size(self.calA, 0), 0, GL_RGBA, GL_FLOAT, self.calA)

        glBindTexture(GL_TEXTURE_2D, self.GLcalB)
        gl_enable_filtering()
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, np.size(self.calB, 1), np.size(self.calB, 0), 0, GL_RGBA, GL_FLOAT, self.calB)

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

    def set_dir(self, dx, dy):
        self.dir_vector = (dx, dy);
        glUniform2f(self.dirLoc, self.dir_vector[0], self.dir_vector[1])
    
    def set_array(self, array):     #sets the array which will be drawn (numpy array passed as a 2d array)
                                    #valid array values are from 0 to 1
        width = np.size(array, 0)
        height = np.size(array, 1)

        if self.shader == None:     #if no shader is used, we are in basic display mode. Only real valued float arrays are accepted
            self.data = array
            self.disable_filter()
            glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, np.size(self.data, 1), np.size(self.data, 0), 0, GL_RED, GL_FLOAT, self.data)
        else:                       #if we are using a shader, we must load the array as a complex floating point texture
            self.data = np.zeros((width, height, 2), 'float')
            self.data[..., 0] = np.real(array)
            self.data[..., 1] = np.imag(array)
            self.enable_filter()
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, np.size(self.data, 1), np.size(self.data, 0), 0, GL_RG, GL_FLOAT, self.data)

    def enable_filter(self):
        glBindTexture(GL_TEXTURE_2D, self.texture)
        gl_enable_filtering()

    def disable_filter(self):
        glBindTexture(GL_TEXTURE_2D, self.texture)
        gl_disable_filtering()

    def enable_blazed(self):
        self.add_shader("vertex.glsl", "blazed.glsl")

    def disable_blazed(self):
        self.shader = None

    def on_draw(self):              #pyglet on draw function, called by pyglet
        self.clear()
        if self.shader == None:
            glEnable(GL_TEXTURE_2D)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.texture)
        else:
            self.shader.use()

            glEnable(GL_TEXTURE_2D)
            
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.texture)

            glEnable(GL_TEXTURE_1D)

            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_1D, self.GL_sinc_lut)

            glActiveTexture(GL_TEXTURE2)
            glBindTexture(GL_TEXTURE_2D, self.GLcalA)

            glActiveTexture(GL_TEXTURE3)
            glBindTexture(GL_TEXTURE_2D, self.GLcalB)

        x = self.x
        y = self.y
        w = self.w
        h = self.h
        pyglet.graphics.draw_indexed(4, pyglet.gl.GL_TRIANGLES, [0, 1, 2, 0, 2, 3],
        ('v2f', (x, y, x+w, y, x+w, y+h, x, y+h)),
        ('t2f', (0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0)))

        if self.shader != None:
            self.shader.clear()
