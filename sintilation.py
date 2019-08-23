import PCAM
import numpy as np
import MathUtils as mu

import time

cam = PCAM.PCAM()

cam.start()

num_of_sample = 100

H_samples = np.zeros(num_of_sample)
V_samples = np.zeros(num_of_sample)
D_samples = np.zeros(num_of_sample)
A_samples = np.zeros(num_of_sample)

for n in range(num_of_sample):
    cam.fetch_buffer()
    H, V, D, A = cam.get_pol()
    center_x, center_y = mu.center_cam(V, np.max(V)*0.7)
    H_samples[n] = mu.circular_integral(H, center_x, center_y, 10)
    V_samples[n] = mu.circular_integral(V, center_x, center_y, 10)
    D_samples[n] = mu.circular_integral(D, center_x, center_y, 10)
    A_samples[n] = mu.circular_integral(A, center_x, center_y, 10)
    cam.queue_buffer()

H_mean, H_std_dev = mu.data_norm(H_samples)
V_mean, V_std_dev = mu.data_norm(V_samples)
D_mean, D_std_dev = mu.data_norm(D_samples)
A_mean, A_std_dev = mu.data_norm(A_samples)

print("H std dev: %f, mean: %f, ratio: %f\n" % (H_std_dev, H_mean, H_std_dev/H_mean))
print("V std dev: %f, mean: %f, ratio: %f\n" % (V_std_dev, V_mean, V_std_dev/V_mean))
print("D std dev: %f, mean: %f, ratio: %f\n" % (D_std_dev, D_mean, D_std_dev/D_mean))
print("A std dev: %f, mean: %f, ratio: %f\n" % (A_std_dev, A_mean, A_std_dev/A_mean))

cam.__del__()