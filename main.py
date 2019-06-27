import numpy as np
import pyglet
import SLM

slm = SLM.SLM(1)
slm2 = SLM.SLM(3)

def update(dt):
    img = np.random.choice([0, 1], size = (64, 64), p=[0.9, 0.1])
    slm.set_location_center(slm.screen_width/2, slm.screen_height/2, 512, 512)
    slm.set_array(img)
    slm2.set_location_center(slm2.screen_width/2, slm2.screen_height/2, 512*2, 512*2)
    slm2.set_array(img)

pyglet.clock.schedule_interval(update, 1/30.0)

pyglet.app.run()