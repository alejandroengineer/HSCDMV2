import numpy as np
import PCAM
import SLM

def retrieve_phase(slm, cam, max_iter = 1000):
    slm_width = slm.screen_width
    slm_height = slm.screen_height

    phase_profile = np.ones((slm_width, slm_height))

    

    return phase_profile