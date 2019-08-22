import PCAM
import AutoUtils as au
import matplotlib.pyplot as plt

cam = PCAM.PCAM()
cam.start()

au.automatic_exposure_and_framing(cam, 400, 3200, 500)

fig=plt.figure(figsize=(2, 2))

cam.fetch_buffer()
H, V, D, A = cam.get_pol()

fig.add_subplot(2, 2, 1)
plt.imshow(H)

fig.add_subplot(2, 2, 2)
plt.imshow(V)

fig.add_subplot(2, 2, 3)
plt.imshow(D)

fig.add_subplot(2, 2, 4)
plt.imshow(A)

cam.queue_buffer()

cam.__del__()

plt.show()