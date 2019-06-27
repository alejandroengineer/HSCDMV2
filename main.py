import numpy as np
import pyglet
import SLM
import PCAM

pcam = PCAM.PCAM()

slm = SLM.SLM(1)
slm2 = SLM.SLM(3)

pcam.set_pixel_format('Mono12')
pcam.set_offset(0, 0)
pcam.set_size(256, 256)

pcam.start()

def update(dt):
    img = np.random.choice([0, 1], size = (64, 64), p=[0.9, 0.1])
    slm.set_location_center(slm.screen_width/2, slm.screen_height/2, 512, 512)
    slm.set_array(img)
    slm2.set_location_center(slm2.screen_width/2, slm2.screen_height/2, 512*2, 512*2)
    data = pcam.fetch_buffer()
    slm2.set_array(data/255)
    pcam.queue_buffer()

pyglet.clock.schedule_interval(update, 1/60.0)

pyglet.app.run()

print('hey')