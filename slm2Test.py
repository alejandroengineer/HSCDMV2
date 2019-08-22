import numpy as np
import SLM2
import glfw

SLM2.init()

slm = SLM2.SLM(3)
slm2 = SLM2.SLM(2)

slm.enable_blazed()

x = np.linspace(-1, 1, 128*4)
y = np.linspace(-1, 1, 128*4)
xx, yy = np.meshgrid(x, y)

dist = np.sqrt((xx**2) + (yy**2))
img = dist < 1

slm.set_array(img.astype(float))

min_size = min(slm.screen_height, slm.screen_width)
slm.set_location_center(slm.screen_width/2, slm.screen_height/2, min_size, min_size)

slm.draw()
slm.swap_buffers()

while not glfw.window_should_close(slm.window):
    slm2.set_array(np.zeros((16, 16)))
    slm2.draw()
    slm2.swap_buffers()
    slm2.set_array(np.ones((16, 16)))
    slm2.draw()
    slm2.swap_buffers()
    glfw.poll_events()

glfw.terminate()
