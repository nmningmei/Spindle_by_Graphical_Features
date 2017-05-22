# -*- coding: utf-8 -*-
"""
Created on Sun May 21 13:13:26 2017

@author: ning
"""

import pandas as pd
import os
import numpy as np
try:
    function_dir = 'D:\\NING - spindle\\Spindle_by_Graphical_Features'
    os.chdir(function_dir)
except:
    function_dir = 'C:\\Users\\ning\\OneDrive\\python works\\Spindle_by_Graphical_Features'
    os.chdir(function_dir)
import eegPipelineFunctions
try:
    file_dir = 'D:\\NING - spindle\\training set\\road_trip\\'
    os.chdir(file_dir)
except:
    file_dir = 'C:\\Users\\ning\\Downloads\\road_trip\\'
    os.chdir(file_dir)
    
for directory_1 in os.listdir(file_dir):
    sub_dir = file_dir + directory_1 + '\\'
    os.chdir(sub_dir)
    for sub_fold in os.listdir(sub_dir):
        sub_fold_dir = sub_dir + sub_fold + '\\'
        os.chdir(sub_fold_dir)
        print(os.listdir(os.getcwd()))
    

        
