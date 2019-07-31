import numpy as np
import pyglet
import SLM
import PCAM
import time
import MathUtils as mu

from instrumental import instrument, list_instruments

slm = SLM.SLM(3)
slm2 = SLM.SLM(2)

slm.enable_blazed()

#slm.load_calibration("H1_cal.mat")

fps_ = 60

pmillis = int(round(time.time() * 1000))

num_of_frames = 0

inst = list_instruments()

print(inst)

#cam = instrument(inst[0])

#cam.master_gain = 1
#cam.gain_boost = False


#cam.start_live_video()

#print(cam.exposure_time)

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
    x = np.linspace(-1, 1, 128*4)
    y = np.linspace(-1, 1, 128*4)
    xx, yy = np.meshgrid(x, y)
    #frame_ready = cam.wait_for_frame()
    #if frame_ready:
    #    img2 = cam.latest_frame()/255#mu.LG(512, 512, 10, 6)#np.random.choice([0, 1], size = (16, 16), p=[0.9, 0.1])#
    #img = mu.LG(512, 512, 10, 6)
    dist = (xx**2) + (yy**2)
    img = dist < 1.0
    #img = img*(1.0 - dist)
    min_size = min(slm.screen_height, slm.screen_width)
    slm.set_location_center(slm.screen_width/2, slm.screen_height/2, min_size, min_size)
    slm.set_array(img)
    slm.enable_filter()
    min_size = min(slm2.screen_height, slm2.screen_width)
    slm2.set_location_center(slm2.screen_width/2, slm2.screen_height/2, min_size, min_size)
    #slm2.set_array(img2)

pyglet.clock.schedule_interval(update, 1/60.0)

event_loop = pyglet.app.EventLoop()

@event_loop.event
def on_window_close(window):
    event_loop.exit()

pyglet.app.run()