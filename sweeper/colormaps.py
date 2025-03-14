import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as plt_colors
from matplotlib import colormaps

def define_phaseColorMap():
    # all numbers from Igor wave 'phaseindex'
    # Igor colors take RGB from 0 to 65535
    rgb = np.zeros((360,3), dtype=np.float64)
    rgb[0:90,0] = np.arange(0, 63000, 700)
    rgb[90:180, 0] = 63000 * np.ones(90)
    rgb[180:270, 0] = np.arange(63000, 0, -700)
    rgb[90:180, 1] = np.arange(0, 63000, 700)
    rgb[180:270, 1] = 63000 * np.ones(90)
    rgb[270:360, 1] = np.arange(63000, 0, -700)
    rgb = rgb  / 65535.0
    # ListedColormap takes an arry of RGB weights normalized to be in [0,1]
    phase_cmap = plt_colors.ListedColormap(rgb, name='phase')
    colormaps.register(phase_cmap, name='phase') 
    

define_phaseColorMap()