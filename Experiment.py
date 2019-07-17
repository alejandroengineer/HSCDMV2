import numpy as np
import pyglet
import SLM
import PCAM
import time
import MathUtils as mu

slm = SLM.SLM(0)

slm.enable_blazed()

fps_ = 60

pmillis = int(round(time.time() * 1000))

num_of_frames = 0

def update(dt):
    global fps_
    global pmillis
    global num_of_frames
    millis = int(round(time.time() * 1000))
    fps_ = 0.9*fps_ + 0.1*(1000/(millis - pmillis))
    if num_of_frames % 60 == 0:
        print(fps_)
    num_of_frames = num_of_frames + 1
    pmillis = millis
    x = np.linspace(0, 1, 16)
    y = np.linspace(0, 1, 16)
    img, yy = np.meshgrid(x, y)
    img = mu.LG(1024, 1024, 10, 6)#np.random.choice([0, 1], size = (16, 16), p=[0.9, 0.1])#
    slm.set_location_center(slm.screen_width/2, slm.screen_height/2, 1024, 1024)
    slm.set_array(img)
    #slm.enable_filter()

pyglet.clock.schedule_interval(update, 1/60.0)

event_loop = pyglet.app.EventLoop()

@event_loop.event
def on_window_close(window):
    event_loop.exit()

pyglet.app.run()