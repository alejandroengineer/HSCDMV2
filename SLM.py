import numpy as np
import pyglet
import scipy.io as sp
from pyglet.gl import *
from pyglet.gl import gl
from pyshaders import from_files_names, ShaderCompilationError
from OpenGL.GL import *
from OpenGL.GLU import *
from MathUtils import *

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

        inv_sinc = np.vectorize(inverse_sinc)

        self.sinc_lut = 1 - inv_sinc(np.linspace(0, 1, 512))

        self.GL_sinc_lut = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.GL_sinc_lut)
        gl_enable_filtering()
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, np.size(self.sinc_lut, 0), 1, 0, GL_RED, GL_FLOAT, self.sinc_lut)

        self.cal_A = np.zeros((64, 64, 4))
        self.cal_A[:, :, 1] = 0.5/np.pi;
        self.GL_cal_A = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.GL_cal_A)
        gl_enable_filtering()
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, np.size(self.cal_A, 1), np.size(self.cal_A, 0), 0, GL_RGBA, GL_FLOAT, self.cal_A)

        self.cal_B = np.zeros((64, 64, 4))
        self.GL_cal_B = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.GL_cal_B)
        gl_enable_filtering()
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, np.size(self.cal_B, 1), np.size(self.cal_B, 0), 0, GL_RGBA, GL_FLOAT, self.cal_B)

        self.Ab_width = int(self.screen_width/4)
        self.Ab_height = int(self.screen_height/4)

        self.Ab_mode = 'fill'
        
        self.GL_cal_Ab = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.GL_cal_Ab)
        gl_enable_filtering()
        self.set_zernike_coeffs([0], [0.5])


    def add_shader(self, vert, frag):
        try:
            self.shader = from_files_names(vert, frag)
            print(self.shader.uniforms)
            self.tex_Loc = glGetUniformLocation(self.shader.pid, 'tex')
            self.cal_A_Loc = glGetUniformLocation(self.shader.pid, 'calA')
            self.cal_B_Loc = glGetUniformLocation(self.shader.pid, 'calB')
            self.cal_Ab_Loc = glGetUniformLocation(self.shader.pid, 'calAb')
            self.lut_Loc = glGetUniformLocation(self.shader.pid, 'Alut')
            self.dir_Loc = glGetUniformLocation(self.shader.pid, 'dir')
            self.screen_size_Loc = glGetUniformLocation(self.shader.pid, 'screen_size')
            self.shader.use()
            glUniform1i(self.tex_Loc, 0)
            glUniform1i(self.cal_A_Loc, 1)
            glUniform1i(self.cal_B_Loc, 2)
            glUniform1i(self.cal_Ab_Loc, 3)
            glUniform1i(self.lut_Loc, 4)
            self.dir_vector = (2.0*np.pi/13.0, 2.0*np.pi/11.0)
            glUniform2f(self.dir_Loc, self.dir_vector[0], self.dir_vector[1])
            glUniform2f(self.screen_size_Loc, self.screen_width, self.screen_height)
            self.shader.clear()
        except ShaderCompilationError as e:
            print(e.logs)
            self.shader = None

    def load_calibration(self, file_name):  #load a calibration file for h/v relations
        calibration = sp.loadmat(file_name)
        self.cal_A = calibration['cal_A']
        self.cal_B = calibration['cal_B']

        glBindTexture(GL_TEXTURE_2D, self.GL_cal_A)
        gl_enable_filtering()
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, np.size(self.cal_A, 1), np.size(self.cal_A, 0), 0, GL_RGBA, GL_FLOAT, self.cal_A)

        glBindTexture(GL_TEXTURE_2D, self.GL_cal_B)
        gl_enable_filtering()
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, np.size(self.cal_B, 1), np.size(self.cal_B, 0), 0, GL_RGBA, GL_FLOAT, self.cal_B)

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
        glUniform2f(self.dir_Loc, self.dir_vector[0], self.dir_vector[1])

    def set_zernike_coeffs(self, pCoeffs, aCoeffs = [1]): #set zernike coeffs for aberation correction
        z = zernike_c_profile(self.Ab_width, self.Ab_height, pCoeffs, aCoeffs, mode = self.Ab_mode)

        self.cal_Ab = np.zeros((self.Ab_height, self.Ab_width, 2), 'float')
        self.cal_Ab[..., 0] = np.real(z)
        self.cal_Ab[..., 1] = np.imag(z)
        glBindTexture(GL_TEXTURE_2D, self.GL_cal_Ab)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, np.size(self.cal_Ab, 1), np.size(self.cal_Ab, 0), 0, GL_RG, GL_FLOAT, self.cal_Ab)
    
    def set_array(self, array):     #sets the array which will be drawn (numpy array passed as a 2d array)
                                    #valid array values are from 0 to 1 if in normal mode, or magnitudes less than 1 in blazed mode
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

            glActiveTexture(GL_TEXTURE0 + 1)
            glBindTexture(GL_TEXTURE_2D, self.GL_cal_A)

            glActiveTexture(GL_TEXTURE0 + 2)
            glBindTexture(GL_TEXTURE_2D, self.GL_cal_B)

            glActiveTexture(GL_TEXTURE0 + 3)
            glBindTexture(GL_TEXTURE_2D, self.GL_cal_Ab)
            
            glActiveTexture(GL_TEXTURE0 + 4)
            glBindTexture(GL_TEXTURE_2D, self.GL_sinc_lut)

        x = self.x
        y = self.y
        w = self.w
        h = self.h
        pyglet.graphics.draw_indexed(4, pyglet.gl.GL_TRIANGLES, [0, 1, 2, 0, 2, 3],
        ('v2f', (x, y, x+w, y, x+w, y+h, x, y+h)),
        ('t2f', (0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0)))

        if self.shader != None:
            self.shader.clear()
