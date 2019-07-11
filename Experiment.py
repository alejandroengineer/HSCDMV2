import numpy as np
import pyglet
import SLM
import PCAM
import time

slm = SLM.SLM(1)

slm.add_shader("vertex.glsl", "blazed.glsl")

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

pyglet.clock.schedule_interval(update, 1/60.0)

event_loop = pyglet.app.EventLoop()

@event_loop.event
def on_window_close(window):
    event_loop.exit()

pyglet.app.run()