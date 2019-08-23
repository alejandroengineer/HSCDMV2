from harvesters.core import Harvester
import numpy as np

class PCAM:
    def __init__(self, camID = 0, CTI = 'C:/Program Files/MATRIX VISION/mvIMPACT Acquire/bin/x64/mvGenTLProducer.cti'):
        self.camID = camID
        self.H = Harvester()
        self.H.add_cti_file(CTI)
        self.H.update_device_info_list()
        self.ia = self.H.create_image_acquirer(camID)
        self.x = self.ia.device.node_map.OffsetX.value
        self.y = self.ia.device.node_map.OffsetY.value
        self.w = self.ia.device.node_map.Width.value
        self.h = self.ia.device.node_map.Height.value
        self.bh = self.ia.device.node_map.BinningHorizontal.value
        self.bv = self.ia.device.node_map.BinningVertical.value
        self.sensor_width = self.ia.device.node_map.SensorWidth.value
        self.sensor_height = self.ia.device.node_map.SensorHeight.value
        self.format = self.ia.device.node_map.PixelFormat.value
        self.exposure = self.ia.device.node_map.ExposureTimeRaw.value
        self.buffer = None
        self.raw_image = None
        self.Hi = None
        self.Vi = None
        self.Di = None
        self.Ai = None

    def __del__(self):
        if self.buffer != None:
            self.buffer.queue()
        self.ia.stop_image_acquisition()
        self.ia.destroy()
        self.H.reset()
    
    def set_pixel_format(self, format):
        self.format = format
        self.ia.device.node_map.PixelFormat.value = format

    def set_size(self, w, h):
        self.w = w
        self.h = h
        self.ia.device.node_map.Width.value = w
        self.ia.device.node_map.Height.value = h
    
    def set_offset(self, x, y):
        self.x = x
        self.y = y
        self.ia.device.node_map.OffsetX.value = x
        self.ia.device.node_map.OffsetY.value = y

    def set_exposure(self, exposure):               #sets exposure time in uS
        self.exposure = exposure
        self.ia.device.node_map.ExposureMode.value = 'Timed'
        self.ia.device.node_map.ExposureTimeRaw.value = exposure

    def set_binning(self, bh = 'x1', bv = 'x1'):    #kind of useless for polarcam
        self.bh = bh
        self.bv = bv
        self.ia.device.node_map.BinningHorizontal.value = bh
        self.ia.device.node_map.BinningVertical.value = bv

    def start(self):
        self.ia.start_image_acquisition()
    
    def stop(self):
        self.ia.stop_image_acquisition()

    def restart(self):
        self.stop()
        self.start()

    def fetch_buffer(self):
        self.buffer = self.ia.fetch_buffer()
        self.raw_image = self.buffer.payload.components[0].data.reshape(self.h, self.w)
        return self.raw_image

    def get_pol(self):
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

    def queue_buffer(self):
        self.buffer.queue()
        self.buffer = None