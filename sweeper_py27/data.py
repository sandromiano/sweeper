import pickle
import os
import time

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))
            
def create_dir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs((dirpath))

def save_data(data, path):
    
    f = open(path, 'wb')
    pickle.dump(data, f)
    f.close() 
        
def load_data(path):
    
    f = open(path, 'rb')
    data = pickle.load(f)
    f.close()
    return(data)

def today():
        
    ct = time.localtime()
    today = 'M' + '{:02d}'.format(ct.tm_mon) + \
            'D' + '{:02d}'.format(ct.tm_mday) + \
            'Y' + '{:04d}'.format(ct.tm_year)
    return(today)
   
def now():
    
    ct = time.localtime()
    now = 'h' + '{:02d}'.format(ct.tm_hour) + \
          'm' + '{:02d}'.format(ct.tm_min) + \
          's' + '{:02d}'.format(ct.tm_sec)
    return(now)

class data_util(object):
    
    def __init__(self, wd = 'C:/data/'):
        '''    
        Parameters
        ----------
        wd : string
            Working Directory. The default is 'C:/data/'.           
        '''
        if wd[-1] != '/':
            wd += '/'
        self.__wd = wd     
    
    @property
    def today_path(self):
        
        folder = self.__wd + today() + '/'
        return(folder)
    
    def sweep_path(self, sweep_type):
        
        return(self.today_path + sweep_type + '_sweep/')
    
    def create_data_folder(self, sweep_type, temp = True):

        folder = self.sweep_path(sweep_type) + now() + '/'
        create_dir(folder)
        if temp:
            create_dir(folder + 'temp/')
        return(folder)
    
    def available_data(self):
        
        list_files(self.main_path)