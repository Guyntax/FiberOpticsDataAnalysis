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



# If merge is true, it will conbine the two dataset and convert the time to hours
# if False, it will simply conver the data to


offset = 0.537

merge = False
convert = True
offset = False

folderPath =  r'C:\Users\d.blach-lafleche\Desktop\Photobleaching_original_dataSets\10m_30kRad\\'
fileNameData = [#['6378LF,10m,60Krad_J0B4_set1',19.54,,0,'60 kRad',True,True], # 6 da
                ['6378LF,10m,30Krad_14-Jun-19_17-32',19.54,17.191,120.058,'30 kRad',True,False]  #% 5 days 

                ]


#################################################################

def MergeDataSet(folderPath,fileNameData,merge):
    if merge == True:
        data = [[],[]]
    else:
        data = [[]]
    
    for j in range(0,len(fileNameData)):
        
        print(j)
        
        fileName = fileNameData[j][0]
        maxPower = fileNameData[j][1]
        minPower = fileNameData[j][2]
        pastTime = fileNameData[j][3]
        radiation = fileNameData[j][4]
        getFit = fileNameData[j][5]
        PremadeDf = fileNameData[j][6]
        
        if PremadeDf:
            data[j] = getTxtPremadeDataFrame(folderPath,fileName)
        else:
            data[j] = getTxtDataFrameWithTimes(folderPath,fileName,pastTime)
    if merge == True:    
        dataset = np.concatenate((data[0],data[1])).tolist()
    else:
        dataset = data[0].tolist()
       
    
    file1 = open("myfile.txt","w")
    for i  in range(0,len(dataset)):
        file1.write("{} ".format(str(dataset[i][0])) + "{} ".format(str(dataset[i][1])) + "\n") 
    file1.close()
    return dataset

###############################################################################################


def offsetData(folderPath,fileNameData,offset):

    fileName = fileNameData[0][0]
    pastTime = fileNameData[0][3]
    PremadeDf = fileNameData[0][6]
    
    if PremadeDf:
        df = getTxtPremadeDataFrame(folderPath,fileName)
    else:
        df = getTxtDataFrameWithTimes(folderPath,fileName,pastTime)
    
    for i in range(0,len(df)):
        df[i][1] = df[i][1] - offset
        
    file1 = open("myfile.txt","w")
    for i  in range(0,len(df)):
        file1.write("{} ".format(str(df[i][0])) + "{} ".format(str(df[i][1])) + "\n") 
    file1.close() 

###############################################################################################
    
if merge:
    dataset = MergeDataSet(folderPath,fileNameData,True)
if convert:
    MergeDataSet(folderPath,fileNameData,False)  
if offsetData:    
    offsetData(folderPath,fileNameData,offset)
 