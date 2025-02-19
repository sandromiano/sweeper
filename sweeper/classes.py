import shutil
import numpy as np
from functools import reduce
from collections import OrderedDict
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
from .data import data_util, save_data, load_data, create_dir
import progressbar

try:
    def define_phaseColorMap():
        # all numbers from Igor wave 'phaseindex'
        # Igor colors take RGB from 0 to 65535
        rgb = np.zeros((360,3), dtype=np.float)
        rgb[0:90,0] = np.arange(0, 63000, 700)
        rgb[90:180, 0] = 63000 * np.ones(90)
        rgb[180:270, 0] = np.arange(63000, 0, -700)
        rgb[90:180, 1] = np.arange(0, 63000, 700)
        rgb[180:270, 1] = 63000 * np.ones(90)
        rgb[270:360, 1] = np.arange(63000, 0, -700)
        rgb = rgb  / 65535.0
        # ListedColormap takes an arry of RGB weights normalized to be in [0,1]
        phase_cmap = plt_colors.ListedColormap(rgb, name='phase')
        plt.register_cmap(name='phase', cmap=phase_cmap) 
        
    define_phaseColorMap()
except:
    pass

class ndsweeps(data_util):

    def __init__(self, wd = 'C:/data/'):
        
        super().__init__(wd = wd)
        self.__state_dict = {}
        self.__axes = OrderedDict()
        self.__update = {}
        self.__action = {}
        self.__AXES = {}
        self.__acquisitions = {}
        self.__data = {}
                
    @property 
    def state_dict(self):
        
        return(self.__state_dict)
    
    @property 
    def axes(self):
        
        return(self.__axes)
    
    @property
    def data(self):
        
        return(self.__data)
    
    @property
    def AXES(self):
        
        return(self.__AXES)
    
    @property
    def acquisitions(self):
        
        return(self.__acquisitions)
    
    def add_state_entry(self, name, value):
        
        self.__state_dict[name] = value

    def add_acquisition(self, name, acquisition = {'None' : lambda: None}):
        
        self.__acquisitions[name] = acquisition

    def add_ax(self, name, values, action):
        
        if not isinstance(values, np.ndarray) or values.ndim != 1:
            raise ValueError('ax values must be a 1-D numpy array.')
        
        self.__axes[name] = np.array(values)
        self.__action[name] = action

    def __build(self):
        
        #builds sweep type name
        axes_names = list(self.__axes.keys())
        self.__sweep_type = '_'.join(axes_names)
        
        #creates meshgrids for each ax
        __NDAXES = np.meshgrid(*self.__axes.values(), indexing = 'ij')
        #shape, to be used to reshape data
        self.__shape = __NDAXES[0].shape 
        #total number of acquisitions
        self.__N = reduce(lambda x, y: x * y, self.__shape)

        #creates 1DAXES dict elements by flattening NDAXES dict element
        for NDAX, key in zip(__NDAXES, self.__axes):
            #values are flattened
            values = NDAX.ravel()
            #defines a "changed" mask, true if value[i] != value[i-1]
            #this is needed to perform "action" only when value changed
            changed = np.concatenate(([True], np.diff(values) != 0))
            #assembles 1DAXES dict
            self.__AXES[key] = {'values' : values,
                                'changed' : changed, 
                                'action' : self.__action[key]}
            
        for name, acquisition in self.__acquisitions.items():
            
            self.__data[name] = {}
            
            for trace_name in acquisition:
                
                self.__data[name][trace_name] = []

    def single_iteration(self, i, save_temp):
        
        temp_axes = {} #temp axes dict
        self.__bar.update(i)
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
                    temp_data[acq_name][trace_name] = self.__data[acq_name][trace_name][-1]
            
            if save_temp: #saves temp_data dicitonary if requested
                temp_trace_folder = self.__folder + 'temp/' + acq_name
                temp_name =  temp_trace_folder + '/' + str(i) + '.pkl'
                save_data(temp_data, temp_name)
                
    def fill_with_NaN(self, i):
        
        for acq_name in self.__data:
            
            for trace_name in self.__data[acq_name]:
                
                single_element = self.__data[acq_name][trace_name][0]
                empty_element = single_element * np.NaN
                self.__data[acq_name][trace_name] += [empty_element for j in range (self.__N - len(self.__data[acq_name][trace_name]))]
                
    def handle_exception(self, i):
        
        user_input = ''
        while user_input not in ['Y', 'N']:
            user_input = input('\n' + \
                               '### SWEEP ABORTED ###\n' + \
                                   'Save temp data? (Y/N): ').upper()
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
    
class dataplot(object):
    
    def __init__(self):
        pass
    
    def set_folder(self, folder):
        self.__folder = folder
    
    def load_data(self):
        
        #loads data
        self.__acquisitions = load_data(self.__folder + 'traces.pkl')
        self.__axes = load_data(self.__folder + 'axes.pkl')
        self.__state = load_data(self.__folder + 'state.pkl')
        #list of available axes
        self.__axes_keys = list(self.__axes.keys())
    
    @property
    def axes(self):
        return(self.__axes)
    
    @property
    def acquisitions(self):
        return(self.__acquisitions)
    
    @property
    def state(self):
        return(self.__state)
    
    def get_slice(self, 
                  fixed_params, 
                  acquisition, 
                  xtrace, 
                  ytrace):
        
        '''
        Parameters
        ----------
        fixed_params : dict {'param_name' : param_value}
            dict of parameters and their values at which slice data
        
        acquisition : str
            name of the acquisition at which extract data
        
        xtrace : str
            name of the trace to use for x
        
        ytrace : str
            name of the trace to use for y

        Returns
        -------
        dictionary of sliced data
        '''
        
        fixed_indexes = {}
        actual_fixed_params = {}
        fixed_params_string = ''
        
        for param_name, param_value in zip(list(fixed_params.keys()), 
                                           list(fixed_params.values())):
            
            fixed_indexes[param_name] = np.argmin(np.abs(self.axes[param_name] \
                                                   - param_value))
            
            actual_fixed_params[param_name] = \
                self.axes[param_name][fixed_indexes[param_name]]
            
            fixed_params_string += \
                param_name + '= ' + str(actual_fixed_params[param_name]) + '\t'
        
        #slice along xaxis, for fixed indexes of other parameters
        _slice = tuple([fixed_indexes[key] for key in self.__axes_keys])
        data = self.acquisitions[acquisition]
        
        if xtrace is not None:
            xname = xtrace
            xdata = data[xtrace][_slice]
        else:
            xname = ''
            xdata = None
            
        yname = ytrace
        ydata = data[ytrace][_slice]

        sliced_data = {'xname' : xname,
                       'yname' : yname,
                       'xdata' : xdata,
                       'ydata' : ydata}
            
        return(sliced_data, fixed_params_string)
    
    def plot_slice(self, 
                   fixed_params, 
                   acquisition, 
                   xtrace = None, 
                   ytrace = None, 
                   xfunc = None, 
                   yfunc = None,
                   **kwargs):
                
        '''
        Parameters
        ----------
        fixed_params : dict {'param_name' : param_value}
            dict of parameters and their values at which slice data
        
        acquisition : str
            name of the acquisition at which extract data
        
        xtrace : str
            name of the trace to use for x
        
        ytrace : str
            name of the trace to use for y

        xfunc : callable
            single-argument function to be applied to xtrace

        yfunc : callable
            single-argument function to be applied to ytrace
        
        **kwargs : optional arguments
            passed to plt.plot
        Returns
        -------
        axis of generated plot
        '''
        
        if ytrace is None:
            raise(ValueError('"ytrace" cannot be "None".'))
        
        data, fixed_params_string = self.get_slice(fixed_params = fixed_params,
                                                   acquisition = acquisition,
                                                   xtrace = xtrace, 
                                                   ytrace = ytrace)
        
        fig, ax = plt.subplots()
        fig.suptitle(fixed_params_string.expandtabs(), wrap = True)
        
        if xtrace is not None:
            xname = data['xname']
            if xfunc is None: 
                xdata = data['xdata']
            else:
                xdata = xfunc(data['xdata'])
                xname = xfunc.__name__ + '(' + xname + ')'
        else:
            xname = ''
            
        yname = data['yname']
        ydata = data['ydata']
        
        if yfunc is not None:
            ydata = yfunc(ydata)
            yname = yfunc.__name__ + '(' + yname + ')'
            
        if xtrace is not None:
            ax.plot(xdata, ydata, **kwargs)
            ax.set_xlabel(xname)
            ax.set_ylabel(yname)
        else:     
            ax.plot(ydata, **kwargs)
            ax.set_ylabel(yname)
        
        plt.tight_layout()
        
        return(ax)
    
    def get_2dslice(self, 
                    xname, 
                    fixed_params,
                    acquisition,
                    ytrace,
                    ztrace):

        '''
        Parameters
        ----------
        xname : string
            name of x axis along which extract data

        fixed_params : dict {'param_name' : param_value}
            dict of parameters and their values at which slice data

        acquisition : str
            name of the acquisition at which extract data

        ytrace : str
            name of the trace to use for y

        ztrace : str
            name of the trace to use for z

        Returns
        -------
        dict = {'xname' : xname,
                'yname' : ytrace,
                'zname' : ztrace,
                'xdata' : x,
                'ydata' : y,
                'zdata' : z}
        '''

        fixed_indexes = {}
        actual_fixed_params = {}
        fixed_params_string = ''
        
        for param_name, param_value in zip(list(fixed_params.keys()), 
                                           list(fixed_params.values())):
            
            fixed_indexes[param_name] = np.argmin(np.abs(self.axes[param_name] \
                                                   - param_value))
            
            actual_fixed_params[param_name] = \
                self.axes[param_name][fixed_indexes[param_name]]
            
            fixed_params_string += \
                param_name + '= ' + str(actual_fixed_params[param_name]) + '\t'
        
        #slice along x axis, for fixed indexes of other parameters
        _slice = tuple([slice(None) if key == xname
                        else fixed_indexes[key] for key in self.__axes_keys])

        data = self.__acquisitions[acquisition]
        yshape = data[ytrace].shape[-1]
        
        x = np.repeat(self.__axes[xname][:, np.newaxis], yshape, axis = 1)
        
        sliced_data = {'xname' : xname,
                       'yname' : ytrace,
                       'zname' : ztrace,
                       'xdata' : x,
                       'ydata' : data[ytrace][_slice],
                       'zdata' : data[ztrace][_slice]}

        return(sliced_data, fixed_params_string)

    def plot_2dslice(self,
                    fixed_params,
                    acquisition, 
                    xparam = None,
                    ytrace = None,
                    ztrace = None,
                    xfunc = None,
                    yfunc = None, 
                    zfunc = None,
                    transpose = False,
                    **kwargs):

        '''
        Parameters
        ----------
        
        fixed_params : dict {'param_name' : param_value}
            dict of parameters and their values at which slice data
        
        acquisition : str
            name of the acquisition from which extract data

        xparam : string
            name of x axis along which extract data
        ytrace : str
            name of the trace to use for y
        
        ztrace : str
            name of the trace to use for z

        xfunc : callable
            single-argument function to be applied to ytrace

        yfunc : callable
            single-argument function to be applied to ytrace

        zfunc : callable
            single-argument function to be applied to ztrace
        
        transpose : boolean (False)
            swaps x and y

        **kwargs : optional arguments
            passed to plt.pcolormesh

        Returns
        -------
        axis of generated plot
        '''
        
        data, fixed_params_string = self.get_2dslice(xname = xparam,
                                                     fixed_params = fixed_params,
                                                     acquisition = acquisition,
                                                     ytrace = ytrace,
                                                     ztrace = ztrace)
    
        if not transpose:
            xdata = data['xdata']
            xname = data['xname']
            ydata = data['ydata']
            yname = data['yname']

        else:
            xdata = data['ydata']
            xname = data['yname']
            ydata = data['xdata']
            yname = data['xname']
        
        zdata = data['zdata']
        zname = data['zname']
    
        if xfunc is not None:
            xdata = xfunc(xdata)
            xname = xfunc.__name__ + '(' + xname + ')'
        if yfunc is not None:
            ydata = yfunc(ydata)
            yname = yfunc.__name__ + '(' + yname + ')'
        if zfunc is not None:
            zdata = zfunc(zdata)
            zname = zfunc.__name__ + '(' + zname + ')'
        
        fig, ax = plt.subplots()
        fig.suptitle(fixed_params_string.expandtabs(), wrap = True)
        
        p = ax.pcolormesh(  xdata, 
                            ydata, 
                            zdata,
                            **kwargs)

        ax.set_xlabel(xname)
        ax.set_ylabel(yname)
        
        cbar = plt.colorbar(p)
        cbar.ax.set_ylabel(zname)
        
        plt.tight_layout()
        
        return(ax, cbar)
    
    def get_2dslice_reduced(self, 
                            xname,
                            yname,
                            fixed_params,
                            acquisition,
                            ztrace,
                            reduce_func):
        
        '''
            Parameters
            ----------
            xname : str
                name of x axis along which extract data

            yname : str
                name of y axis along which extract data
            
            fixed_params : dict {'param_name' : param_value}
                dict of parameters and their values at which slice data
            
            acquisition : str
                name of the acquisition at which extract data
            
            ztrace : str
                name of the trace to use for z

            reduce_func : callable
                single-argument function which acts on last index of the ztrace to reduce
                array dimension from 3 to 2. Typically used to average repeated measurements.
            Returns
            -------
            dict = {'xname' : xname,
                    'yname' : ytrace,
                    'zname' : ztrace,
                    'xdata' : x,
                    'ydata' : y,
                    'zdata' : reduce_func(z)}
            '''
        
        fixed_indexes = {}
        
        for param_name, param_value in zip(list(fixed_params.keys()), 
                                           list(fixed_params.values())):
            
            fixed_indexes[param_name] = np.argmin(np.abs(self.axes[param_name] \
                                                   - param_value))
        
        #slice along x axis, for fixed indexes of other parameters
        _slice = tuple([slice(None) if key == xname or key == yname
                        else fixed_indexes[key] for key in self.__axes_keys])

        data = self.__acquisitions[acquisition]
        
        x, y = np.meshgrid(self.__axes[xname], self.__axes[yname])
        
        sliced_data = {'xname' : xname,
                       'yname' : yname,
                       'zname' : ztrace,
                       'xdata' : x,
                       'ydata' : y,
                       'zdata' : reduce_func(data[ztrace][_slice])}
            
        return(sliced_data)            
    
    def plot_2dslice_reduced(self,
                            fixed_params,
                            acquisition, 
                            xparam = None,
                            yparam = None,
                            ztrace = None,
                            reduce_func = lambda x : np.mean(x, axis = -1),
                            xfunc = None,
                            yfunc = None, 
                            zfunc = None,
                            transpose = False,
                            **kwargs):
        
        '''
            Parameters
            ----------
            
            fixed_params : dict {'param_name' : param_value}
                dict of parameters and their values at which slice data
            
            acquisition : str
                name of the acquisition at which extract data

            xparam : str
                name of x axis along which extract data

            yparam : str
                name of y axis along which extract data

            ztrace : str
                name of the trace to use for z

            reduce_func : callable
                single-argument function which acts on last index of the ztrace to reduce
                array dimension from 3 to 2. Typically used to average repeated measurements.
            Returns
            -------
            dict = {'xname' : xname,
                    'yname' : ytrace,
                    'zname' : ztrace,
                    'xdata' : x,
                    'ydata' : y,
                    'zdata' : reduce_func(z)}
            '''
        
        data = self.get_2dslice_reduced(xname = xparam, 
                                        yname = yparam,
                                        fixed_params = fixed_params,
                                        acquisition = acquisition,
                                        ztrace = ztrace,
                                        reduce_func = reduce_func)
         
        if not transpose:
            xdata = data['xdata']
            xname = data['xname']
            ydata = data['ydata']
            yname = data['yname']

        else:
            xdata = data['ydata']
            xname = data['yname']
            ydata = data['xdata']
            yname = data['xname']
        
        zdata = data['zdata']
        zname = data['zname']
         
        if xfunc is not None:
            xdata = xfunc(xdata)
            xname = xfunc.__name__ + '(' + xname + ')'
        if yfunc is not None:
            ydata = yfunc(ydata)
            yname = yfunc.__name__ + '(' + yname + ')'
        if zfunc is not None:
            zdata = zfunc(zdata)
            zname = zfunc.__name__ + '(' + zname + ')'
         
        fig, ax = plt.subplots()
         
        p = ax.pcolormesh(xdata, 
                        ydata, 
                        zdata,
                        **kwargs)
    
        ax.set_xlabel(xname)
        ax.set_ylabel(yname)

        cbar = plt.colorbar(p)
        cbar.ax.set_ylabel(zname)

        plt.tight_layout()

        return(ax)
                
    def interactive_plot(self, 
                         acquisition = None, 
                         trace = None, 
                         bins = 50,
                         lim = 0.5):
        '''
        Experimental - not documented yet
        '''
        acquisition = self.acquisitions[acquisition]
        xdata = acquisition[trace].real
        ydata = acquisition[trace].imag
        zeros = tuple([0 for ax in self.axes])
        hist0 = np.histogram2d(xdata[zeros], ydata[zeros], bins = bins,
                               density = True, 
                               range = [[-lim, lim],[-lim, lim]])
        
        # Create the figure and the line that we will manipulate
        
        fig, ax = plt.subplots()
        quad = ax.pcolormesh(hist0[1], hist0[2], hist0[0])
    
        ax.set_xlabel('Re')
        ax.set_xlabel('Im')
        ax.set_aspect(1)
        plt.axhline(0)
        plt.axvline(0)
        # adjust the main plot to make room for the sliders
        plt.subplots_adjust(left=0.25, bottom=0.3)
    
        self.sliders = []
    
        for i, ax in enumerate(self.axes):
            
            sl = Slider(
                ax = plt.axes([0.25, 0.1 + i * 0.05, 0.65, 0.03]),
                label = ax,
                valmin = self.axes[ax][0],
                valmax = self.axes[ax][-1],
                valstep = self.axes[ax],
                valinit = self.axes[ax][0],
            )
            self.sliders.append(sl)
    
        # The function to be called anytime a slider's value changes
        def update(val):
            indexes = tuple([np.where(self.axes[ax] == slider.val)[0][0] 
                       for ax, slider in zip(self.axes, self.sliders)])
            hist = np.histogram2d(xdata[indexes], 
                                   ydata[indexes], 
                                   bins = bins, 
                                   density = True,
                                   range = [[-lim, lim],[-lim, lim]])
            
            quad.set_array(hist[0])
            fig.canvas.draw_idle()
    
        for slider in self.sliders:
            # register the update function with each slider
            slider.on_changed(update)
    
        # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    
        def reset(event):
            for slider in self.sliders:
                slider.reset()
    
        resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Reset', hovercolor='0.975')
        button.on_clicked(reset)
    
        plt.show()
        
    def interactive_1dplot(self,
                           acquisition = None, 
                           xtrace = None, 
                           ytrace = None, 
                           xfunc = None,
                           yfunc = None,
                           ylim = None,
                           **kwargs):
        
        '''
        Experimental - not documented yet
        '''

        acquisition = self.acquisitions[acquisition]
        
        zeros = tuple([0 for ax in self.axes])
        
        xdata = acquisition[xtrace]
        ydata = acquisition[ytrace]
        
        if xfunc is not None:
            xdata = xfunc(xdata)
            
        if yfunc is not None:
            ydata = yfunc(ydata)
        
        # Create the figure and the line that we will manipulate
        fig, ax = plt.subplots()
        plot, = ax.plot(xdata[zeros], ydata[zeros], **kwargs)
        
        if ylim is not None:
            ax.set_ylim(ylim)
    
        # adjust the main plot to make room for the sliders
        plt.subplots_adjust(left = 0.25, bottom = 0.3)
    
        self.sliders = []
    
        slider_axes = self.axes
    
        for i, ax in enumerate(slider_axes):
            
            if np.diff(slider_axes[ax])[0] < 0:
                slider_axes[ax] = np.flip(slider_axes[ax])
                
            sl = Slider(
                ax = plt.axes([0.25, 0.1 + i * 0.05, 0.65, 0.03]),
                label = ax,
                valmin = slider_axes[ax][0],
                valmax = slider_axes[ax][-1],
                valstep = slider_axes[ax],
                valinit = slider_axes[ax][0],
            )
            self.sliders.append(sl)
    
        # The function to be called anytime a slider's value changes
        def update(val):
            indexes = tuple([np.where(slider_axes[ax] == slider.val)[0][0] 
                       for ax, slider in zip(slider_axes, self.sliders)])
            
            plot.set_data(xdata[indexes], ydata[indexes])
            fig.canvas.draw_idle()
    
        for slider in self.sliders:
            # register the update function with each slider
            slider.on_changed(update)
    
        # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    
        def reset(event):
            for slider in self.sliders:
                slider.reset()
    
        resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Reset', hovercolor='0.975')
        button.on_clicked(reset)
        
        plt.show()