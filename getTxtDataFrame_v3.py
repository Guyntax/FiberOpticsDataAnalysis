import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf
from sklearn.metrics import r2_score


#folderPath =  r'C:\Users\d.blach-lafleche\Desktop\photobleaching- files\\'
folderPath =  r'C:\Users\d.blach-lafleche\Desktop\photobleaching\\'
fileName = '6387LF,10m,60Krad__06-Nov-18_18-42'
#fileName = 'Coractive-ER8-6,7.5m,60 krad_05-Mar-19_19-24_Repaired'
#fileName = 'Coractive-SCF-ER60.2m,60 krad'    #No days given
#fileName = 'ER60_2m,100 krad_19-Dec-18_17-40'

def getTimeInterval(time1,time2):
    #*************************************************************************#
    #* This function take time in the format hour:minute:second and returns  *#
    #* he difference in between the two times                                *#
    #*************************************************************************#
    
    time1 = time1.split(':')
    time2 = time2.split(':')
    
    time1 = (3600*float(time1[0])+
            60*float(time1[1])+
            float(time1[2]))/3600 #Convert to house
    time2=  (3600*float(time2[0])+
            60*float(time2[1])+
            float(time2[2]))/3600 #Convert to house
    
    interval = time2-time1
    return interval


def getTxtDataFrame(folderPath,fileName):
    #*************************************************************************#
    #* This function reads a file with a time stamp and power and return a   *#
    #* an array with time in hours and it corresponding power, but the time  *#
    #* interval must be constant
    #*************************************************************************#    
    
    path = folderPath + fileName + '.txt'
    file = open(path)
    #print(file.readlines())
    
    df = file.readlines()
    data = np.ones((len(df)-1,2))
    
    for i in range(0,len(df)):
        tempData = df[i].split()
        #print(tempData)
        if len(tempData) == 5:
            #print(5)
            if i == 1:
                time1 = tempData[1]
                data[i-1,0] = 1e-50
                #print(tempData)
                data[i-1,1] = tempData[3]
            elif i==2:
                time2 = tempData[1]
                interval = getTimeInterval(time1,time2)
                data[i-1,1] = tempData[3]
                data[i-1,0] = interval
                #print(interval)
            elif i>2 and tempData != []:
                data[i-1,0] = data[i-2,0] + interval
                data[i-1,1] = tempData[3]
        elif len(tempData) == 4:
            #print(4)
            if i == 1:
                time1 = tempData[0]
                data[i-1,0] = 1e-50
                #print(tempData[2])
                data[i-1,1] = tempData[2]
            elif i==2:
                time2 = tempData[0]
                interval = getTimeInterval(time1,time2)
                #print(tempData[2])
                data[i-1,1] = tempData[2]
                data[i-1,0] = interval
                #print(interval)
            elif i>2 and tempData != []:
                data[i-1,0] = data[i-2,0] + interval
                #print(tempData[2])
                data[i-1,1] = tempData[2]           
            
    return data

data = getTxtDataFrame(folderPath,fileName)


def getTxtPremadeDataFrame(folderPath,fileName):
    #*************************************************************************#
    #* This function reads a .txt file with time in hours and power, and     *#
    #* and returns the array as is                                           *#
    #*************************************************************************#
    
    print((folderPath + fileName + '.txt'))
    file = open(folderPath + fileName + '.txt',"r")
    
    df = file.readlines()
    for i in range(0,len(df)):
#        print(df[i])
        df[i] = (df[i].split())
        df[i][0] = float(df[i][0])
        #print(i)
        df[i][1] = float(df[i][1])
        
    df =np.array(df)    
    
    return df

def offsetData(df,offset):
    # offsets y dimension of data (power)
    for i in range(0,len(df)):
        df[i][1] = df[i][1] - offset
        
    return df


#df = getTxtPremadeDataFrame(folderPath,fileName)