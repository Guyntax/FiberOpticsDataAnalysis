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
from getTxtDataFrame_v3 import getTxtPremadeDataFrame
from getTxtDataFrameWithTimes_v3  import getTxtDataFrameWithTimes
from getTxtDataFrameWithTimes_v3  import getTimes
from getTxtDataFrameWithTimes_v3  import get24Hour

#*****************************************************************************#
#* Title: PhotoBleaching Analysis                                            *#
#* Author: Didier Blach-Lalfèche                                             *#
#* Date: August 16th 2019                                                    *#
#* Company: MPBC                                                             *#
#* Description: This script is for prepareing the raw data by accouting for  *#
#*              pumping time and according to the pumping calendat as well as*#
#*              the power offset between jigs.                               *#
#*              The files in the Order list will be read and  the time       *#
#*              between them will be the time specified as pumping time.     *#
#*                                                                           *#
#* Other script nessecary: getTxtDataFrame_v3 (Contains functions)           *#
#*                         getTxtDataFrameWithTimes_v3 (Contains functions)  *#
#*****************************************************************************#

offset = 0.537

merge = True
convert = False
offset = False


folderPath =  [r'C:\Users\d.blach-lafleche\Desktop\Photobleaching_Temp\\',
               r'C:\Users\d.blach-lafleche\Desktop\Photobleaching_original_dataSets\10m_30kRad\\',
               r'C:\Users\d.blach-lafleche\Desktop\Photobleaching_original_dataSets\10m_60kRad\\',
               r'C:\Users\d.blach-lafleche\Desktop\Photobleaching_original_dataSets\10m_100kRad\\',
               r'C:\Users\d.blach-lafleche\Desktop\Photobleaching_original_dataSets\6.5m_100kRad\\',
               ]
folderDestination =  [r'C:\Users\d.blach-lafleche\Desktop\Photobleaching\\',
               r'C:\Users\d.blach-lafleche\Documents\Python Scripts\Photobleaching_v3\30kRad\\',
               r'C:\Users\d.blach-lafleche\Documents\Python Scripts\Photobleaching_v3\60kRad\\',
               r'C:\Users\d.blach-lafleche\Documents\Python Scripts\Photobleaching_v3\100kRad\\',
               r'C:\Users\d.blach-lafleche\Documents\Python Scripts\Photobleaching_v3\6.5m\\',
               ]







## 30 kRad
#Order = [[1,'data','6387LF,10m,30Krad__01-Nov-18_17-44',False,-0.14],
#         [1,'no pumping',0],
#         [1,'pumping',264.368], 
#         [1,'data','6378LF,10m,30krad_J0B3_25-26-27 Juin',False,0.04],#0.185 #0.08
#         [1,'pumping',547.64], 
#         [1,'data','6378LF,10m,30kRad_J0B1_19-Jul-19_10-54',False,0],
#         [1,'data','6378LF,10m,30kRad_J0B1_23-Jul-19_16-35',False,0],
#         ]

## 60 kRad
#Order = [[2,'data','6387LF,10m,60Krad__06-Nov-18_18-42',False,-0.14],
#         [2,'no pumping',0], 
#         [2,'data','6378LF,10m,60Krad_J0B4_14-Jun-19_18-41',False,0.6],
#         [2,'data','6378LF,10m,60Krad_J0B4_17-Jun-19_17-49 (2)',False,0.6],
#         [2,'pumping',385.123-80], 
#         [2,'data','6378LF,10m,60kRad,J0B4_05-Jul-19_14-55',False,0.6-0.22],
##         [2,'data','6378LF,10m,60kRad_J0B4_09-Jul-19_09-16',False,0.537],
#         [2,'pumping',49.7343], 
#         [2,'data','6378LF,10m,60kRad_J0B1_11-Jul-19_12-54',False,0],
#         ]

## 100 kRad
#Order = [[3,'data','6378LF,10m,100Krad_12-Nov-18_16-50_corrected',False,0],
#         [3,'no pumping',0], 
#         [3,'data','6378LF,10m,100krad_06-Jun-19_16-39_croped',False,0],
#         [3,'pumping',216.673], 
#         [3,'data','6378LF,10m,100krad,J0B1_21-Jun-19_12-42',False,0],
#         [3,'pumping',363.613], 
#         [3,'data','6378LF,10m,100kRad_J0B4_09-Jul-19_09-16',False,0],
#         [3,'data','6378LF,10m,100kRad_J0B4_09-Jul-19_09(2)-16',False,0],
#        ]

## 6.5m
Order = [[4,'data','6378LF,6.5m,100Krad,INP_20-Mar-18_18-24_recovered',False,0],
         [4,'no pumping',24], 
         [4,'data','6378LF,6.5m,100kRad_J0B2_14-Jun-19_18-10',False,0.72],
         [4,'pumping',334.411], 
         [4,'data','6378LF,6.5m,100kRad_J0B2_28-Jun-19_17-00(corrected)',False,0.72],
         [4,'no pumping',48], 
         [4,'data','6378LF,6.5m,100kRad_J0B2_02-Jul-19_09-23',False,0.72],
         [4,'data','6378LF,6.5m,100kRad_J0B2_02-Jul-19_09-23(2)',False,0.72],
         [4,'data','6378LF,6.5m,100kRad_JoB2_03-Jul-19_09-24',False,0.72],
         [4,'data','6378LF,6.5m,100kRad_JoB2_03-Jul-19_09-24(2)',False,0.7],
         [4,'pumping',597.904],
         [4,'data','6378LF,6.5m,100kRad_J0B2_30-Jul-19_12-40',False,0.72],
         ]

data = [[]]*(len(Order)-4)







def offsetData(df,offset):
    for i in range(0,len(df)):
        df[i][1] = df[i][1] - offset  
    return df
###############################################################################################
    



j = 0
pastTime = 0
for i in range(0,len(Order)):
    
#    print(pastTime)
    if Order[i][1] == 'data':
        path = folderPath[Order[i][0]]
        destination = folderDestination[Order[i][0]]
        fileName = Order[i][2]
        PremadeDf = Order[i][3]
        offset = Order[i][4]
        if PremadeDf:
            data[j] = getTxtPremadeDataFrame(path,fileName)
        else:
            data[j] = getTxtDataFrameWithTimes(path,fileName,0)
        
        if offset != 0:
            data[j] = offsetData(data[j],offset)
        
        data[j] = np.transpose(data[j])
        
        
        data[j][:][0] = data[j][:][0] + pastTime
        
        data[j] = np.transpose(data[j])
        
            
#        print(data[j][-1][0])
        pastTime =  data[j][-1][0]

        j = j+1
        
    elif Order[i][1] == 'pumping':
        pastTime = pastTime + Order[i][2]
        



dataset = np.transpose(np.array([[],[]]))
for i in range(0,len(data)):
    dataset = np.concatenate((dataset,data[i]))   
dataset = np.array(dataset)

file = open(destination + "CombinedData.txt","w")
for i  in range(0,len(dataset)):
    file.write("{} ".format(str(dataset[i][0])) + "{} ".format(str(dataset[i][1])) + "\n") 
file.close() 


for j in range(0,len(data)):
    file = open(destination + "6378LF,10m,100kRad_set-{}.txt".format(j+1),"w")
    for i in range(0,len(data[j])):
        file.write("{} ".format(str(data[j][i][0])) + "{} ".format(str(data[j][i][1])) + "\n") 
    file.close() 





        
        