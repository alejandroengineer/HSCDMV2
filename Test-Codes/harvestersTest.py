from harvesters.core import Harvester
import numpy as np

h = Harvester()

h.add_cti_file('C:/Program Files/MATRIX VISION/mvIMPACT Acquire/bin/x64/mvGenTLProducer.cti')

h.update_device_info_list()

ia = h.create_image_acquirer(0)

ia.device.node_map.Width.value, ia.device.node_map.Height.value = 256, 256
ia.device.node_map.OffsetX.value, ia.device.node_map.OffsetY.value = 256, 256

ia.device.node_map.PixelFormat.value = 'Mono12'

