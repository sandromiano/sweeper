import numpy as np
import matplotlib.pyplot as plt
from ninatool.circuits.base_circuits import  snail
from ninatool.internal.structures import Nlosc
from ninatool.internal.tools import unitsConverter
from copy import copy
from functools import reduce

uc = unitsConverter()

#EXTERNAL AXES
flux = np.linspace(0,1,101) * 2 * np.pi
power = np.linspace(-20,0,11)
#WANNABE INTERNAL AX
span = np.linspace(-100e8,100e8,51)

FLUX, SPAN = np.meshgrid(flux, span, indexing = 'ij')

sn0 = snail()
sn0.interpolate_results(flux)
snosc0 = Nlosc(sn0, name = 'snosc')

#INTERNAL AXIS
FREQ = (SPAN.T + snosc0.omega * uc.frequency_units).T
#%%

axes = {'freq' : {'values' : FREQ, 'action' : lambda x : print(x), 'external' : True},
        'flux' : {'values' : FLUX, 'action' : lambda x : print(x), 'external' : False}}

is_external = np.array([int(ax['external']) for ax in axes.values()])

NDIM = len(axes)
NINT = np.sum(is_external)

SHAPE = FLUX.shape

INTERNAL_SHAPE = SHAPE[:NDIM-1]
EXTERNAL_SHAPE = SHAPE[NDIM-1:]

INTERNAL_N = reduce(lambda x, y: x * y, INTERNAL_SHAPE)

_slice = tuple([slice(None) if x==0 else 0 for x in is_external])
#%%

flat_axes = copy(axes)

if NINT != 0:
    for ax in axes:
        if axes[ax]['external'] == True:
            flat_axes[ax]['values'] = np.reshape(axes[ax]['values'], (INTERNAL_N,) + EXTERNAL_SHAPE)
        else:
            flat_axes[ax]['values'] = np.ravel(flat_axes[ax]['values'][_slice])
#%%
plt.figure()
plt.plot(FLUX, FREQ)