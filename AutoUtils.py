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

def automatic_phase(cam, slm, start = 0, end = 0.5, num_of_samples = 32, depth = 0):
    max_depth = 2

    if depth >= max_depth:
        return np.pi*(start+end)/2

    b_mag = 1

    slm.set_centered_size(slm.screen_width, slm.screen_height)

    b_values = []

    for i in range(num_of_samples):
        phase = start + (i*(end - start)/(num_of_samples - 1))
        slm.set_array(np.ones((64, 64))*np.exp(1.0j*np.pi*phase))
        slm.draw()
        slm.swap_buffers()

        time.sleep(0.025)

        cam.fetch_avg()
        H, V, D, A = cam.get_pol()
        H, V, D, A = cam.get_pol()
        center_x, center_y = mu.center_cam(V, np.max(V)*0.7)
        V_value = mu.circular_integral(V, center_x, center_y, 5)
        H_value = mu.circular_integral(H, center_x, center_y, 5)/V_value
        D_value = mu.circular_integral(D, center_x, center_y, 5)/V_value
        A_value = mu.circular_integral(A, center_x, center_y, 5)/V_value
        V_value = 1

        HVtoDA = (H_value+V_value)/(D_value+A_value)

        a, b = mu.solveDM(H_value, V_value, HVtoDA*D_value, HVtoDA*A_value, np.pi*0.1)

        b_mag = np.absolute(b)

        b_values.append(b_mag)

        print((b_mag, phase))

    min_phase_loc = b_values.index(min(b_values))

    best_phase = start + (min_phase_loc*(end - start)/(num_of_samples - 1))

    print(best_phase)

    span = (end - start)/num_of_samples

    return automatic_phase(cam, slm, best_phase - 2*span, best_phase + 2*span, 64, depth+1)

def automatic_slm_center(cam, slm, slm2, size, size_ratio, phase, phase_low = 0.16*np.pi, h_mask_res = 4):
    slm.set_array(0.75*mu.diamond(128))
    slm.set_centered_size(size, size)
    slm.draw()
    slm.swap_buffers()

    time.sleep(0.1)

    slm2_size = size*size_ratio
    scan_size = slm2_size/4

    num_regions_x = int(np.floor(slm2.screen_width/scan_size)+1)
    num_regions_y = int(np.floor(slm2.screen_height/scan_size)+1)

    automatic_exposure_and_framing(cam, 800, 3400*16, 200*16)

    slm.set_array(0.0*np.ones((64, 64)))
    slm.set_centered_size(size, size)
    slm.draw()
    slm.swap_buffers()

    cam.get_darkframe()

    slm.set_array(0.75*mu.diamond(128))
    slm.set_centered_size(size, size)
    slm.draw()
    slm.swap_buffers()

    phase_low = automatic_phase(cam, slm2)

    slm2.set_zero(np.exp(1.0j*phase_low))

    a_values = np.zeros((num_regions_x, 1), 'complex')
    b_values = np.zeros((num_regions_x, 1), 'complex')

    slm2.set_array(np.ones((64, 64))*np.exp(1.0j*(phase_low+phase)))

    for i in range(num_regions_x):
        slm2.set_location(i*scan_size, 0, scan_size, slm2.screen_height)
        slm2.draw()
        slm2.swap_buffers()
        time.sleep(0.025)
        cam.fetch_avg()
        H, V, D, A = cam.get_pol()
        center_x, center_y = mu.center_cam(V, np.max(V)*0.7)
        V_value = mu.circular_integral(V, center_x, center_y, 5)
        H_value = mu.circular_integral(H, center_x, center_y, 5)/V_value
        D_value = mu.circular_integral(D, center_x, center_y, 5)/V_value
        A_value = mu.circular_integral(A, center_x, center_y, 5)/V_value
        V_value = 1

        HVtoDA = (H_value+V_value)/(D_value+A_value)

        a, b = mu.solveDM(H_value, V_value, HVtoDA*D_value, HVtoDA*A_value, phase)

        a_values[i][0] = a
        b_values[i][0] = b

        cam.queue_buffer()

    b_magnitudes = np.absolute(b_values)
    max_b_value = np.max(b_magnitudes)
    x, y = mu.center_cam(b_magnitudes, max_b_value*0.2)

    x_out = (x+0.5)*scan_size

    for i in range(num_regions_y):
        slm2.set_location(0, i*scan_size, slm2.screen_width, scan_size)
        slm2.draw()
        slm2.swap_buffers()
        time.sleep(0.025)
        cam.fetch_avg()
        H, V, D, A = cam.get_pol()
        center_x, center_y = mu.center_cam(V, np.max(V)*0.7)
        V_value = mu.circular_integral(V, center_x, center_y, 5)
        H_value = mu.circular_integral(H, center_x, center_y, 5)/V_value
        D_value = mu.circular_integral(D, center_x, center_y, 5)/V_value
        A_value = mu.circular_integral(A, center_x, center_y, 5)/V_value
        V_value = 1

        HVtoDA = (H_value+V_value)/(D_value+A_value)

        a, b = mu.solveDM(H_value, V_value, HVtoDA*D_value, HVtoDA*A_value, phase)

        a_values[i][0] = a
        b_values[i][0] = b

        cam.queue_buffer()

    b_magnitudes = np.absolute(b_values)
    max_b_value = np.max(b_magnitudes)
    x, y = mu.center_cam(b_magnitudes, max_b_value*0.2)

    y_out = (x+0.5)*scan_size

    print((x_out, y_out))

    slm2.set_location_center(x_out, y_out, slm2_size*1.25, slm2_size*1.25)

    return x_out, y_out, phase_low

def automatic_slm_center2(cam, slm, slm2, size, size_ratio, phase, phase_low = 0.16*np.pi, h_mask_res = 4):
    slm.set_array(0.75*np.ones((64, 64)))
    slm.set_centered_size(size, size)
    slm.draw()
    slm.swap_buffers()

    time.sleep(0.1)

    slm2_size = size*size_ratio
    scan_size = slm2_size/4

    num_regions_x = int(np.floor(slm2.screen_width/scan_size)+1)
    num_regions_y = int(np.floor(slm2.screen_height/scan_size)+1)

    automatic_exposure_and_framing(cam, 800, 3400*16, 200*16)

    slm.set_array(0.0*np.ones((64, 64)))
    slm.set_centered_size(size, size)
    slm.draw()
    slm.swap_buffers()

    cam.get_darkframe()

    slm.set_array(0.75*np.ones((64, 64)))
    slm.set_centered_size(size, size)
    slm.draw()
    slm.swap_buffers()

    phase_low = automatic_phase(cam, slm2)

    slm2.set_zero(np.exp(1.0j*phase_low))

    a_values = np.zeros((num_regions_x, num_regions_y), 'complex')
    b_values = np.zeros((num_regions_x, num_regions_y), 'complex')
    d_values = np.zeros((num_regions_x, num_regions_y))

    slm2.set_array(np.ones((64, 64))*np.exp(1.0j*(phase_low+phase)))

    for i in range(num_regions_x):
        for j in range(num_regions_y):
            slm2.set_location(i*scan_size, j*scan_size, scan_size, scan_size)
            slm2.draw()
            slm2.swap_buffers()
            time.sleep(0.025)
            cam.fetch_avg()
            H, V, D, A = cam.get_pol()
            center_x, center_y = mu.center_cam(V, np.max(V)*0.7)
            V_value = mu.circular_integral(V, center_x, center_y, 5)
            H_value = mu.circular_integral(H, center_x, center_y, 5)/V_value
            D_value = mu.circular_integral(D, center_x, center_y, 5)/V_value
            A_value = mu.circular_integral(A, center_x, center_y, 5)/V_value
            V_value = 1

            HVtoDA = (H_value+V_value)/(D_value+A_value)

            a, b = mu.solveDM(H_value, V_value, HVtoDA*D_value, HVtoDA*A_value, phase)

            a_values[i][j] = a
            b_values[i][j] = b
            d_values[i][j] = A_value

            cam.queue_buffer()

    b_magnitudes = np.absolute(b_values)
    max_b_value = np.max(b_magnitudes)
    x, y = mu.center_cam(b_magnitudes, max_b_value*0.7)

    x = (x+0.5)*scan_size
    y = (y+0.5)*scan_size

    print((x, y))

    slm2.set_location_center(x, y, slm2_size*1.25, slm2_size*1.25)
    #slm2.set_location(0, 0, slm2.screen_width, slm2.screen_height)

    region_size = h_mask_res
    num_of_samples = region_size**2

    hadamard = mu.hadamard_masks(region_size)
    hadamard = hadamard.astype('float')

    b_vector = np.zeros(num_of_samples, 'complex')

    cam.fetch_buffer()
    H, V, D, A = cam.get_pol()
    center_x, center_y = mu.center_cam(V, np.max(V)*0.7)
    cam.queue_buffer()

    for i in range(num_of_samples):
        slm2.set_array(np.exp(1.0j*(phase_low + phase*np.reshape(hadamard[i], (region_size, region_size)))))
        slm2.disable_filter()
        slm2.draw()
        slm2.swap_buffers()
        time.sleep(0.025)
        cam.fetch_avg()
        H, V, D, A = cam.get_pol()
        V_value = mu.circular_integral(V, center_x, center_y, 5)
        H_value = mu.circular_integral(H, center_x, center_y, 5)/V_value
        D_value = mu.circular_integral(D, center_x, center_y, 5)/V_value
        A_value = mu.circular_integral(A, center_x, center_y, 5)/V_value
        V_value = 1

        D_value2 = D_value

        cam.queue_buffer()

        HVtoDA = (H_value+V_value)/(D_value+A_value)

        a1, b1 = mu.solveDM(H_value, V_value, D_value*HVtoDA, A_value*HVtoDA, phase)

        slm2.set_array(np.exp(1.0j*(phase_low + phase*np.reshape(1.0 - hadamard[i], (region_size, region_size)))))
        slm2.disable_filter()
        slm2.draw()
        slm2.swap_buffers()
        time.sleep(0.025)
        cam.fetch_avg()
        H, V, D, A = cam.get_pol()
        center_x, center_y = mu.center_cam(V, np.max(V)*0.7)
        V_value = mu.circular_integral(V, center_x, center_y, 5)
        H_value = mu.circular_integral(H, center_x, center_y, 5)/V_value
        D_value = mu.circular_integral(D, center_x, center_y, 5)/V_value
        A_value = mu.circular_integral(A, center_x, center_y, 5)/V_value
        V_value = 1

        HVtoDA = (H_value+V_value)/(D_value+A_value)

        print(HVtoDA)

        a2, b2 = mu.solveDM(H_value, V_value, D_value*HVtoDA, A_value*HVtoDA, phase)

        b_vector[i] = b1 - b2

        cam.queue_buffer()

    hadamard_t = 2.0*hadamard - 1.0

    final_img = np.reshape(np.matmul(hadamard_t, b_vector), (region_size, region_size))

    return x, y, final_img, phase_low

def measure(cam, slm, phase_low, phase, mask, center = None, take_dual = True):
    slm.set_array(np.exp(1.0j*(phase_low + phase*mask)))
    slm.disable_filter()
    slm.draw()
    slm.swap_buffers()

    time.sleep(0.025)    

    cam.fetch_avg()

    H, V, D, A = cam.get_pol()

    center_x, center_y = mu.center_cam(V, np.max(V)*0.7)

    if not (center == None):
        center_x = center[0]
        center_y = center[1]
        
    print((center_x, center_y))

    integral_radius = 3

    V_value = mu.circular_integral(V, center_x, center_y, integral_radius)
    H_value = mu.circular_integral(H, center_x, center_y, integral_radius)/V_value
    D_value = mu.circular_integral(D, center_x, center_y, integral_radius)/V_value
    A_value = mu.circular_integral(A, center_x, center_y, integral_radius)/V_value
    V_value = 1

    HVtoDA = (H_value+V_value)/(D_value+A_value)

    a, b = mu.solveDM(H_value, V_value, D_value*HVtoDA, A_value*HVtoDA, phase)

    if take_dual:
        a2, b2 = measure(cam, slm, phase_low, phase, 1.0 - mask, None, False)
        a = (a - a2)/2
        b = (b - b2)/2

    return a, b