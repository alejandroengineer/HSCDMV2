import os
import PySpin
import numpy as np

class PCAM:
    def __init__(self, camID = 0):
        self.camID = camID
        self.system = PySpin.System.GetInstance()
        cam_list = self.system.GetCameras()
        self.cam = cam_list[camID]
        self.cam.Init()
        self.cam.BeginAcquisition()
        self.cam.EndAcquisition()
        self.nodemap = self.cam.GetNodeMap()

        self.x = self.cam.OffsetX.GetValue()
        self.y = self.cam.OffsetY.GetValue()
        self.w = self.cam.Width.GetValue()
        self.h = self.cam.Height.GetValue()
        self.sensor_width = self.cam.SensorWidth.GetValue()
        self.sensor_height = self.cam.SensorHeight.GetValue()
        self.exposure = 10000
        self.buffer = None
        self.raw_image = None
        self.Hi = None
        self.Vi = None
        self.Di = None
        self.Ai = None

        self.capturing = False

    def __del__(self):  #done
        if self.buffer != None:
            self.buffer.Release()
        if self.capturing:
            self.cam.EndAcquisition()
        self.cam.DeInit()
        del self.cam
        self.system.ReleaseInstance()
    
    def set_pixel_format(self, format = "Polarized16"): #done
        self.format = format
        if self.capturing:
            self.stop()
            node_pixel_format = PySpin.CEnumerationPtr(self.nodemap.GetNode("PixelFormat"))
            node_pixel_format_ = PySpin.CEnumEntryPtr(node_pixel_format.GetEntryByName(format))
            pixel_format_ = node_pixel_format_.GetValue()
            node_pixel_format.SetIntValue(pixel_format_)
            self.start()
        else:
            node_pixel_format = PySpin.CEnumerationPtr(self.nodemap.GetNode("PixelFormat"))
            node_pixel_format_ = PySpin.CEnumEntryPtr(node_pixel_format.GetEntryByName(format))
            pixel_format_ = node_pixel_format_.GetValue()
            node_pixel_format.SetIntValue(pixel_format_)

    def set_size(self, w, h): #done
        self.w = w
        self.h = h
        if self.capturing:
            self.stop()
            self.cam.Width.SetValue(w)
            self.cam.Height.SetValue(h)
            self.start()
        else:
            self.cam.Width.SetValue(w)
            self.cam.Height.SetValue(h)
    
    def set_offset(self, x, y): #done
        self.x = x
        self.y = y
        if self.capturing:
            self.stop()
            self.cam.OffsetX.SetValue(32)
            self.cam.OffsetY.SetValue(32)
            self.start()
        else:
            self.cam.OffsetX.SetValue(32)
            self.cam.OffsetY.SetValue(32)

    def set_exposure(self, exposure): #done         #sets exposure time in uS
        self.exposure = exposure
        if self.capturing:
            self.stop()
            self.cam.ExposureMode.SetIntValue(1)
            self.cam.ExposureTime.SetValue(exposure)
            self.start()
        else:
            self.cam.ExposureMode.SetIntValue(1)
            self.cam.ExposureTime.SetValue(exposure)

    def start(self):    #done
        self.cam.BeginAcquisition()
        self.capturing = True
    
    def stop(self):     #done
        self.cam.EndAcquisition()
        self.capturing = False

    def restart(self):  #done
        self.stop()
        self.start()

    def fetch_buffer(self): #done
        self.buffer = self.cam.GetNextImage()
        self.raw_image = np.array(self.buffer.GetData(), dtype="uint64").reshape( (self.buffer.GetHeight(), self.buffer.GetWidth()) )
        return self.raw_image

    def get_pol(self):  #done
        num = np.size(self.raw_image)
        h = np.size(self.raw_image, 0)
        w = np.size(self.raw_image, 1)

        data = np.reshape(self.raw_image, (num, 1))

        data_1 = data[0:num:2]
        data_2 = data[1:num:2]

        data_1_2d = np.reshape(data_1, (h, int(w/2)))
        data_2_2d = np.reshape(data_2, (h, int(w/2)))

        self.Di = data_1_2d[0::2, :].astype(float)
        self.Vi = data_1_2d[1::2, :].astype(float)
        self.Hi = data_2_2d[0::2, :].astype(float)
        self.Ai = data_2_2d[1::2, :].astype(float)
        
        return self.Hi, self.Vi, self.Di, self.Ai

    def queue_buffer(self): #done
        self.buffer.Release()
        self.buffer = None