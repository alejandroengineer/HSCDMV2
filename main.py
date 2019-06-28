import numpy as np
import pyglet
import SLM
import PCAM
import time

pcam = PCAM.PCAM()

slm = SLM.SLM(1)
slm2 = SLM.SLM(3)

print((pcam.sensor_width, pcam.sensor_height))

pcam.set_pixel_format('Mono12')
pcam.set_offset(292, 92)#73,23)#
pcam.set_size(1024, 1024)
pcam.set_binning('x1', 'x1')
pcam.set_exposure(9200)

print(pcam.exposure)

pcam.start()

fps_ = 60

pmillis = int(round(time.time() * 1000))

def update(dt):
    global fps_
    global pmillis
    millis = int(round(time.time() * 1000))
    fps_ = 0.9*fps_ + 0.1*(1000/(millis - pmillis))
    print(fps_)
    pmillis = millis
    img = np.random.choice([0, 1], size = (64, 64), p=[0.9, 0.1])
    slm.set_location_center(slm.screen_width/2, slm.screen_height/2, 512, 512)
    slm.set_array(img)
    slm2.set_location_center(slm2.screen_width/2, slm2.screen_height/2, 512*2, 512*2)
    data = pcam.fetch_buffer()
    slm2.set_array(data/4096)
    pcam.queue_buffer()

pyglet.clock.schedule_interval(update, 1/60.0)

event_loop = pyglet.app.EventLoop()

@event_loop.event
def on_window_close(window):
    event_loop.exit()

pyglet.app.run()

pcam.__del__()