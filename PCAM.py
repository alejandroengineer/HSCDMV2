from harvesters.core import Harvester
import numpy as np

class PCAM:
    def __init__(self, camID = 0, CTI = 'C:/Program Files/MATRIX VISION/mvIMPACT Acquire/bin/x64/mvGenTLProducer.cti'):
        self.camID = camID
        self.h = Harvester()
        self.h.add_cti_file(CTI)
        self.h.update_device_info_list()
        self.ia = self.h.create_image_acquirer(camID)
    
    def set_pixel_format(self, format):
        self.ia.device.node_map.PixelFormat.value = format
    
    def set_frame_rate(self, rate):
        self.ia.device.node_map.

    def start(self):

