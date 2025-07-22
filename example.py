import numpy as np
from time import sleep
from sweeper.classes import ndsweeps, dataplot

                              ### --- ###
                           ###ACQUISITION###
                              ### --- ###
def random_w_sleep(x):
    sleep(0.1)
    return(np.random.random(x))

def set_fstart(x):
    pass
def set_fstop(x):
    pass
def set_npoints(x):
    pass

def set_frequency_ax(x):
    set_fstart(x[0])
    set_fstop(x[-1])
    set_npoints(len(x))
    
#initializes a ndsweeps instance
swp = ndsweeps(wd = 'C:/data/sweeper_example/')

#adds p1 ax
p1 = np.linspace(-1,1,11)

swp.add_ax(name = 'p1',
           values = p1,
           action = lambda x : print('\n p1 = ' + str(x)))

#adds p2 ax
p2 = np.linspace(-1,1,11)

swp.add_ax(name = 'p2',
           values = p2,
           action = lambda x : print('p2 = ' + str(x)))

#adds acquisition
swp.add_acquisition(name = 'LINEAR_RESPONSE',
                    internal_ax = swp.ax(name = 'freq1',
                                         action = set_frequency_ax,
                                         values = np.linspace(1,2,101)),
                    acquisition = {'S11' : lambda : random_w_sleep(11),
                                   'S21' : lambda : random_w_sleep(11)},)

swp.add_acquisition(name = 'STARK_SHIFT',
                    internal_ax = swp.ax(name = 'freq2',
                                         action = set_frequency_ax,
                                         values = np.linspace(1,2,101)),
                    acquisition = {'S11' : lambda : random_w_sleep(11),
                                   'S21' : lambda : random_w_sleep(11)},)
#runs the sweep and retrieves the path to data
folder = swp.run()
#%%
                              ### --- ###
                          ###VISUALIZATION###
                              ### --- ###
#initializes a dataplot instance
swp_plot = dataplot()
#sets folder to retrieve data
swp_plot.set_folder(folder)
#loads data from folder
swp_plot.load_data()
#%%
#plots trace (x and y) for certain indexes of parameters

def dB(x):
    return(10 * np.log10(np.abs(x)))

swp_plot.plot_slice(fixed_params = {'p1' : -1,
                                    'p2' : -1},
                    acquisition = 'S11',
                    xtrace = 'FREQ',
                    ytrace = 'POL',
                    xfunc = dB,
                    yfunc = np.real)

swp_plot.plot_slice(fixed_params = {'p1' : -1,
                                    'p2' : -1},
                    acquisition = 'S21',
                    xtrace = 'FREQ',
                    ytrace = 'POL',
                    xfunc = dB,
                    yfunc = np.real)
#%%

#plot 2d slice
swp_plot.plot_2dslice(fixed_params = {'p2' : 0.5},
                    acquisition = 'S11',
                    xparam = 'p1',
                    ytrace = 'FREQ',
                    ztrace = 'POL',
                    yfunc = dB,
                    zfunc = np.real)