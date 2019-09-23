import os
import PySpin

import matplotlib.pyplot as plt
import numpy as np

fig=plt.figure(figsize=(1, 1))

system = PySpin.System.GetInstance()

cam_list = system.GetCameras()

num_cameras = cam_list.GetSize()

print('number of cameras: %d\n' % num_cameras)

if num_cameras > 0:
    cam = cam_list[0]

    cam.Init()

    cam.BeginAcquisition()
    cam.EndAcquisition()

    nodemap = cam.GetNodeMap()

    #set camera settings
    node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
    node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
    acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
    node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

    cam.Width.SetValue(100)
    cam.Height.SetValue(100)

    cam.OffsetX.SetValue(32)
    cam.OffsetY.SetValue(32)

    if cam.PixelFormat.GetAccessMode() == PySpin.RW:
        print('able to change the format\n')
        print(cam.PixelFormat.GetValue())
        node_pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode("PixelFormat"))
        node_pixel_format_mono8 = PySpin.CEnumEntryPtr(node_pixel_format.GetEntryByName("Polarized16"))
        pixel_format_mono8 = node_pixel_format_mono8.GetValue()
        node_pixel_format.SetIntValue(pixel_format_mono8)
    else:
        print('not able to change the format\n')

    cam.GainAuto.SetValue(PySpin.GainAuto_Off)
    cam.Gain.SetValue(1.0)

    cam.ExposureMode.SetIntValue(1)
    cam.ExposureTime.SetValue(10000)

    cam.AdcBitDepth.SetIntValue(2)

    print((cam.SensorWidth.GetValue(), cam.SensorHeight.GetValue()))

    cam.BeginAcquisition()

    for n in range(10):
        acqdFrame = cam.GetNextImage()

        #fig.add_subplot(2, 2, 1)
        data = np.array(acqdFrame.GetData(), dtype="uint64").reshape( (acqdFrame.GetHeight(), acqdFrame.GetWidth()) )

        # plt.imshow(data[::11, ::11], interpolation='nearest')
        # plt.pause(0.05)

        print('\nlargest value: %d\n' %data[5][5])

        print(n)

        acqdFrame.Release()

    cam.EndAcquisition()

    cam.DeInit()

    del cam

else:
    print('too few cameras to run test\n')

cam_list.Clear()

system.ReleaseInstance()

plt.show()