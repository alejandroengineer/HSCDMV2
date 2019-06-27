import numpy as np
import pyglet
import SLM

slm = SLM.SLM(0)
#slm2 = SLM(0)

def update(dt):
    img = np.random.choice([0, 1], size = (64, 64), p=[0.9, 0.1])
    slm.set_location_center(slm.screen_width/2, slm.screen_height/2, 512, 512)
    slm.set_array(img)
    #slm2.set_location(0, 0, 512, 512)
    #slm2.set_array(img)

pyglet.clock.schedule_interval(update, 1/30.0)

pyglet.app.run()