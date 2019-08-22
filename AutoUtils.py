import numpy as np
import time
import SLM2
import PCAM
import MathUtils as mu

def automatic_exposure(cam, target_exposure, exposure_tolerance):
    good_exposure = False
    while good_exposure == False:
        cam.fetch_buffer()
        max_pixel = np.max(cam.raw_image)
        if abs(max_pixel - target_exposure) > exposure_tolerance:
            new_exposure = target_exposure*cam.exposure/max_pixel
            if new_exposure <= 10:
                print("Unable to achive desired exposure: Intensity too high\n")
                break
            if new_exposure >= 12078:
                print("Unable to achive desired exposure: Intensity too low\n")
                break
            cam.set_exposure(int(new_exposure))
        else:
            good_exposure = True
        cam.queue_buffer()
        time.sleep(0.2)
    cam.fetch_buffer()
    max_pixel = np.max(cam.raw_image)
    print("Exposure acheived: %f" % (max_pixel))
    cam.queue_buffer()

def automatic_exposure_and_framing(cam, size, target_exposure, exposure_tolerance):
    cam.set_binning()
    cam.set_pixel_format('Mono12')
    cam.set_offset(int(0), int(0))
    cam.set_size(int(cam.sensor_width), int(cam.sensor_height))
    time.sleep(1)
    automatic_exposure(cam, target_exposure, exposure_tolerance)
    x = 0
    y = 0
    for n in range(10):
        time.sleep(0.2)
        dy, dx = mu.center_cam(cam.fetch_buffer())
        cam.queue_buffer()
        x = x + dx - (size/2)
        y = y + dy - (size/2)
        x = min(max(np.floor(x/2)*2, 0), cam.sensor_width - size)
        y = min(max(np.floor(y/2)*2, 0), cam.sensor_height - size)
        cam.set_size(int(size), int(size))
        cam.set_offset(int(x), int(y))
