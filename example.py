import numpy as np
from time import sleep
from sweeper.classes import ndsweeps, dataplot

                              ### --- ###
                           ###ACQUISITION###
                              ### --- ###
def random_w_sleep(x):
    sleep(1e-6)
    return(np.random.random(x))

def preamble1():
    print('preamble 1 called')
    
def preamble2():
    print('preamble 2 called')    

#initializes a ndsweeps instance
swp = ndsweeps(wd = 'C:/data/sweeper_example/')

#adds p1 ax
p1 = np.linspace(-1,1,21)

swp.add_ax(name = 'p1',
           values = p1,
           action = lambda x : print('\n p1 = ' + str(x)))

#adds p2 ax
p2 = np.linspace(-1,1,21)

swp.add_ax(name = 'p2',
           values = p2,
           action = lambda x : print('p2 = ' + str(x)))

#adds acquisition

swp.add_acquisition(name = 'S11',
                    preamble = preamble1,
                    acquisition = {'POL' : lambda : random_w_sleep(11),
                                   'FREQ' : lambda : random_w_sleep(11)})

swp.add_acquisition(name = 'S21',
                    preamble = preamble2,
                    acquisition = {'POL' : lambda : random_w_sleep(11),
                                   'FREQ' : lambda : random_w_sleep(11)})
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
#%%
swp_plot.plot_slice_reduced(fixed_params = {'p1' : -1},
                            acquisition = 'S11',
                            xparam = 'p2',
                            ytrace = 'POL',
                            xfunc = None,
                            yfunc = np.real,
                            reduce_func = lambda x : x[..., 0])