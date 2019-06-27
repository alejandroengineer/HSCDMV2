from harvesters.core import Harvester
import numpy as np

h = Harvester()

h.add_cti_file('/Users/kznr/dev/genicam/bin/Maci64_x64/TLSimu.cti')

h.update_device_info_list()

ia = h.create_image_acquirer(0)

ia.device.node_map.Width.value, ia.device.node_map.Height.value = 8, 8

ia.device.node_map.PixelFormat.value = 'Mono8'

ia.start_image_acquisition()