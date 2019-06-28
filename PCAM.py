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

    def fetch_buffer(self):
        self.buffer = self.ia.fetch_buffer()
        return self.buffer.payload.components[0].data.reshape(self.w, self.h)

    def queue_buffer(self):
        self.buffer.queue()
        self.buffer = None