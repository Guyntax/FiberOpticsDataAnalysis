#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#from scipy.special import erf
from sklearn.metrics import r2_score
#from getTxtDataFrame import getTimeInterval
#from astropy.table import Table

from getTxtDataFrame_v3 import getTimeInterval
from getTxtDataFrame_v3  import getTxtDataFrame
from getTxtDataFrameWithTimes_v3  import getTxtDataFrameWithTimes
from getTxtDataFrameWithTimes_v3  import getTimes
from getTxtDataFrameWithTimes_v3  import get24Hour


folderPath =  r'C:\Users\d.blach-lafleche\Desktop\Photobleaching\\'

fileNameData = [#['6378LF,10m,100Krad_12-Nov-18_16-50',19.54,16.039,0,'100 kRad',True], # 4 days
                #['6378LF,10m,100krad_06-Jun-19_16-39 - Copy',19.54,16.039,96.5701,'100 kRad',False], # 6 da
                ['6378LF,10m,60Krad_J0B4_14-Jun+17-Juin',19.54,17.191,141.075,'60 kRad',False,False]
                ]





