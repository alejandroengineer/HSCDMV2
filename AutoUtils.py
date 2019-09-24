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
        time.sleep(0.15)
    cam.fetch_buffer()
    max_pixel = np.max(cam.raw_image)
    print("Exposure acheived: %f" % (max_pixel))
    cam.queue_buffer()

def automatic_exposure_and_framing(cam, size, target_exposure, exposure_tolerance):
    cam.set_pixel_format()
    cam.set_exposure(50)
    cam.set_offset(int(0), int(0))
    cam.set_size(int(cam.sensor_width), int(cam.sensor_height))
    time.sleep(0.2)
    cam.restart()
    automatic_exposure(cam, target_exposure, exposure_tolerance)
    x = 0
    y = 0
    for n in range(10):
        time.sleep(0.15)
        dy, dx = mu.center_cam(cam.fetch_buffer())
        cam.queue_buffer()
        x = x + dx - (size/2)
        y = y + dy - (size/2)
        x = min(max(np.floor(x/4)*4, 0), cam.sensor_width - size)
        y = min(max(np.floor(y/4)*4, 0), cam.sensor_height - size)
        cam.set_size(int(size), int(size))
        cam.set_offset(int(x), int(y))

def automatic_phase_noQWP(cam, slm):
    phase = 0

    slm.set_array(np.ones((64, 64))*np.exp(1.0j*phase))
    slm.draw()
    slm.swap_buffers()

    time.sleep(0.1)

    cam.fetch_buffer()
    H, V, D, A = cam.get_pol()
    value = np.sum(D)/np.sum(A)

def automatic_slm_center(cam, slm, slm2, size, size_ratio, phase, phase_low = 0.16*np.pi):
    slm.set_array(0.75*np.ones((64, 64)))
    slm.set_centered_size(size, size)
    slm.draw()
    slm.swap_buffers()

    time.sleep(0.1)

    slm2_size = size*size_ratio
    scan_size = slm2_size/4

    num_regions_x = int(np.floor(slm2.screen_width/scan_size)+1)
    num_regions_y = int(np.floor(slm2.screen_height/scan_size)+1)

    automatic_exposure_and_framing(cam, 800, 3200*16, 200*16)

    a_values = np.zeros((num_regions_x, num_regions_y), 'complex')
    b_values = np.zeros((num_regions_x, num_regions_y), 'complex')
    d_values = np.zeros((num_regions_x, num_regions_y))

    slm2.set_array(np.ones((64, 64))*np.exp(1.0j*phase))

    cam.fetch_buffer()
    cam.queue_buffer()
    cam.fetch_buffer()
    cam.queue_buffer()
    cam.fetch_buffer()
    cam.queue_buffer()
    cam.fetch_buffer()
    cam.queue_buffer()
    cam.fetch_buffer()
    cam.queue_buffer()

    for i in range(num_regions_x):
        for j in range(num_regions_y):
            slm2.set_location(i*scan_size, j*scan_size, scan_size, scan_size)
            slm2.draw()
            slm2.swap_buffers()
            time.sleep(0.025)
            cam.fetch_buffer()
            cam.queue_buffer()
            cam.fetch_buffer()
            cam.queue_buffer()
            cam.fetch_buffer()
            cam.queue_buffer()
            cam.fetch_buffer()
            cam.queue_buffer()
            cam.fetch_buffer()
            cam.queue_buffer()
            cam.fetch_buffer()
            H, V, D, A = cam.get_pol()
            center_x, center_y = mu.center_cam(V, np.max(V)*0.7)
            H_value = mu.circular_integral(H, center_x, center_y, 5)
            V_value = mu.circular_integral(V, center_x, center_y, 5)
            D_value = mu.circular_integral(D, center_x, center_y, 5)
            A_value = mu.circular_integral(A, center_x, center_y, 5)

            a, b = mu.solveDM(H_value, V_value, D_value, A_value, phase - phase_low)

            a_values[i][j] = a
            b_values[i][j] = b
            d_values[i][j] = A_value

            cam.queue_buffer()

    b_magnitudes = np.absolute(b_values)
    max_b_value = np.max(b_magnitudes)
    x, y = mu.center_cam(b_magnitudes, max_b_value*0.5)

    x = (x+0.5)*scan_size
    y = (y+0.5)*scan_size

    print((x, y))

    slm2.set_location_center(x, y, slm2_size*1.25, slm2_size*1.25)
    #slm2.set_location(0, 0, slm2.screen_width, slm2.screen_height)

    region_size = 8
    num_of_samples = region_size**2

    hadamard = mu.hadamard_masks(region_size)
    hadamard = hadamard.astype('float')

    b_vector = np.zeros(num_of_samples, 'complex')

    cam.fetch_buffer()
    H, V, D, A = cam.get_pol()
    center_x, center_y = mu.center_cam(V, np.max(V)*0.7)
    cam.queue_buffer()

    cam.fetch_buffer()
    cam.queue_buffer()
    cam.fetch_buffer()
    cam.queue_buffer()
    cam.fetch_buffer()
    cam.queue_buffer()
    cam.fetch_buffer()
    cam.queue_buffer()
    cam.fetch_buffer()
    cam.queue_buffer()

    for i in range(num_of_samples):
        slm2.set_array(np.exp(1.0j*(phase_low + (phase - phase_low)*np.reshape(hadamard[i], (region_size, region_size)))))
        slm2.disable_filter()
        slm2.draw()
        slm2.swap_buffers()
        time.sleep(0.025)
        cam.fetch_buffer()
        cam.queue_buffer()
        cam.fetch_buffer()
        cam.queue_buffer()
        cam.fetch_buffer()
        cam.queue_buffer()
        cam.fetch_buffer()
        cam.queue_buffer()
        cam.fetch_buffer()
        cam.queue_buffer()
        cam.fetch_buffer()
        H, V, D, A = cam.get_pol()
        H_value = mu.circular_integral(H, center_x, center_y, 4)
        V_value = mu.circular_integral(V, center_x, center_y, 4)
        D_value = mu.circular_integral(D, center_x, center_y, 4)
        A_value = mu.circular_integral(A, center_x, center_y, 4)

        D_value2 = D_value

        cam.queue_buffer()

        HVtoDA = (H_value+V_value)/(D_value+A_value)

        a1, b1 = mu.solveDM(H_value, V_value, D_value*HVtoDA, A_value*HVtoDA, phase - phase_low)

        slm2.set_array(np.exp(1.0j*(phase_low + (phase - phase_low)*np.reshape(1.0 - hadamard[i], (region_size, region_size)))))
        slm2.disable_filter()
        slm2.draw()
        slm2.swap_buffers()
        time.sleep(0.025)
        cam.fetch_buffer()
        cam.queue_buffer()
        cam.fetch_buffer()
        cam.queue_buffer()
        cam.fetch_buffer()
        cam.queue_buffer()
        cam.fetch_buffer()
        cam.queue_buffer()
        cam.fetch_buffer()
        cam.queue_buffer()
        cam.fetch_buffer()
        H, V, D, A = cam.get_pol()
        center_x, center_y = mu.center_cam(V, np.max(V)*0.7)
        H_value = mu.circular_integral(H, center_x, center_y, 4)
        V_value = mu.circular_integral(V, center_x, center_y, 4)
        D_value = mu.circular_integral(D, center_x, center_y, 4)
        A_value = mu.circular_integral(A, center_x, center_y, 4)

        HVtoDA = (H_value+V_value)/(D_value+A_value)

        a2, b2 = mu.solveDM(H_value, V_value, D_value*HVtoDA, A_value*HVtoDA, phase - phase_low)

        b_vector[i] = b1 - b2

        cam.queue_buffer()

    hadamard_t = 2.0*hadamard - 1.0

    final_img = np.reshape(np.matmul(hadamard_t, b_vector), (region_size, region_size))

    return x, y, d_values, final_img, b_values