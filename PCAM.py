from harvesters.core import Harvester
import numpy as np

class PCAM:
    def __init__(self, camID = 0, CTI = 'C:/Program Files/MATRIX VISION/mvIMPACT Acquire/bin/x64/mvGenTLProducer.cti'):
        self.camID = camID
        self.H = Harvester()
        self.H.add_cti_file(CTI)
        self.H.update_device_info_list()
        self.ia = self.H.create_image_acquirer(camID)
        self.x = self.ia.device.node_map.OffsetX
        self.y = self.ia.device.node_map.OffsetY
        self.w = self.ia.device.node_map.Width
        self.h = self.ia.device.node_map.Height

    def __del__(self):
        print('heyy')
        self.buffer.queue()
        print('hey')
        self.ia.stop_image_acquisition()
        print('hey2')
        self.ia.destroy()
        print('hey3')
        self.H.reset()
    
    def set_pixel_format(self, format):
        self.ia.device.node_map.PixelFormat.value = format

    def set_size(self, w, h):
        self.w = w
        self.h = h
        self.ia.device.node_map.Width = w
        self.ia.device.node_map.Height = h
    
    def set_offset(self, x, y):
        self.x = x
        self.y = y
        self.ia.device.node_map.OffsetX = x
        self.ia.device.node_map.OffsetY = y

    def set_exposure(self, exposure):
        self.exposure = exposure
        self.ia.device.node_map.OffsetY

    def start(self):
        self.ia.start_image_acquisition()

    def fetch_buffer(self):
        self.buffer = self.ia.fetch_buffer()
        return self.buffer.payload.components[0].data.reshape(self.w, self.h)

    def queue_buffer(self):
        self.buffer.queue()