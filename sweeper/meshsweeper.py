import shutil
import numpy as np
from numpy.random import random
from functools import reduce
from collections import OrderedDict
from matplotlib import pyplot as plt
from matplotlib import colors as plt_colors
from matplotlib.widgets import Slider, Button
from data import data_util, save_data, load_data, create_dir
import progressbar
from copy import copy
        
class meshsweep(data_util):

    def __init__(self, wd = 'C:/data/'):
        
        super().__init__(wd = wd)
        self.__state = {}
        self.__internal_axis = {}
        self.__update = {}
        self.__actions = {}
        self.__AXES = {}
        self.__acquisition = {}
        self.__data = {}

    def add_AXIS(self, name, values, action):
        
        self.__AXES[name] = np.array(values)
        self.__actions[name] = action

    def __build(self):

        if len(set([AXIS.shape for AXIS in self.__AXES])) > 1:
            raise ValueError('axes are not compatible')
        else:
            #flattens internal AXIS
            self.__internal_AXIS = self.__internal_axis.reshape(-1, len(self.internal_axis[-1]))
            self.__internal_AXIS_changed = np.concatenate([True, np.diff(self.__INTERNAL_AXIS, axis = 0).any(axis = 1)])
            #total number of acquisitions
            self.__N = reduce(lambda x, y: x * y, self.__shape)
    
            #creates 1DAXES dict elements by flattening NDAXES dict element
            for AX in self.__AXES:
                #values are flattened
                values = self.__AXES[AX].ravel()
                #defines a "changed" mask, true if value[i] != value[i-1]
                #this is needed to perform "action" only when value changed
                changed = np.concatenate(([True], np.diff(values) != 0))
                #assembles 1DAXES dict
                self.__AXES[AX] = {'values' : values,
                                    'changed' : changed, 
                                    'action' : self.__actions[AX]}

    def single_iteration(self, i, save_temp):
        
        temp_axes = {} #temp axes dict
        self.__bar.update(i)
        if self.__internal_AXIS_changed[i] == True:
            self.__internal_AXIS_action(self.__internal_AXIS[i])
        #sets swept parameters
        for AX in self.__AXES:
            
                #checks if ax has changed, then performs action
                if self.__AXES[AX]['changed'][i]:
                    axval = self.__AXES[AX]['values'][i]
                    self.__AXES[AX]['action'](axval)
                    
                #inserts current value of ax in temp axes dict
                temp_axes[AX] = self.__AXES[AX]['values'][i]
                                    
        #sweeps traces
        for acq_name, acquisition in self.__acquisitions.items():
            #acquires each trace component
            
            if save_temp: #creates temp_data dictionary if requested
                temp_data = {acq_name : {}}
                temp_data['axes'] = temp_axes
            
            for trace_name, trace_func in acquisition.items():
                
                self.__data[acq_name][trace_name].append(trace_func())  
                if save_temp: #fills temp_data dictionary if requested
                    temp_data[acq_name][trace_name] = \
                        self.__data[acq_name][trace_name][-1]
            
            if save_temp: #saves temp_data dicitonary if requested
                temp_trace_folder = self.__folder + 'temp/' + acq_name
                temp_name =  temp_trace_folder + '/' + str(i) + '.pkl'
                save_data(temp_data, temp_name)
                
    def fill_with_NaN(self, i):
        
        for acq_name in self.__data:
            
            for trace_name in self.__data[acq_name]:
                
                N = len(self.__data[acq_name][trace_name])
                single_element = self.__data[acq_name][trace_name][0]
                empty_element = single_element * np.NaN
                self.__data[acq_name][trace_name] += \
                    [empty_element for j in range (self.__N - N)]
                
    def handle_exception(self, i):
        
        user_input = ''
        while user_input not in ['Y', 'N']:
            user_input = input('Save temp data? (Y/N) ').upper()
        if user_input == 'Y':
            self.fill_with_NaN(i)
        return(user_input)
    
    def reshape_data(self):
        
        for acq_name, acq_data in self.__data.items():
            
            #THIS NEEDS TO BE DONE BETTER, SKETCHY AS IT IS NOW BECAUSE DOESN'T
            #ALLOW DIFFERENT ITERATIONS TO HAVE DIFFERENT SHAPE. BEST IS TO
            #FILL WITH NONE or NaN THE REMAINING SLOTS.
            for trace_name, trace_data in acq_data.items():
                
                if isinstance(trace_data[0], np.ndarray):
                    shape = trace_data[0].shape
                    
                else:
                    shape = ()
                    
                shape = self.__shape + shape
                self.__data[acq_name][trace_name] = np.reshape(trace_data, shape)

    def save_data(self):
        #saves ND traces data
        save_data(self.__data, self.__folder + 'traces.pkl')
        print('\n\ndata was saved in folder: ' + '\'' + self.__folder + '\'')
        
    def run(self, save_temp = True):
        
        self.__build() #builds flattened axes
        
        self.__folder = self.create_data_folder(sweep_type = self.__sweep_type,
                                         temp = save_temp)
        
        if save_temp: #creates temp_folders for each acquisition if requested
            for acq_name in self.__acquisitions:
                temp_acquisition_folder = self.__folder + 'temp/' + acq_name
                create_dir(temp_acquisition_folder)
                
        #saves axes data before starting, useful to reconstruct
        save_data(self.__axes, self.__folder + 'axes.pkl')
        #saves state dictionary, keeps memory of fixed settings
        save_data(self.__state_dict, self.__folder + 'state.pkl')

        self.__bar = progressbar.ProgressBar(max_value = self.__N)
        
        for i in range(self.__N):
            try:
                self.single_iteration(i = i, save_temp = save_temp)
                    
            except KeyboardInterrupt:
                user_input = self.handle_exception(i = i)
                if user_input == 'N':
                    shutil.rmtree(self.__folder)
                    return(None)
                elif user_input == 'Y':
                    break
        
        self.reshape_data()
        self.save_data()
        #removes temp folder if created
        if save_temp:
            shutil.rmtree(self.__folder + 'temp/')
        return(self.__folder)
    
