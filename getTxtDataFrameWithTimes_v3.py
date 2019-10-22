import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf
from sklearn.metrics import r2_score


#folderPath =  r'C:\Users\d.blach-lafleche\Desktop\photobleaching- files\\'
folderPath =  r'C:\Users\d.blach-lafleche\Desktop\photobleaching\\'
folderPath = r'C:\Users\d.blach-lafleche\Desktop\Photobleaching_original_dataSets\6.5m_100kRad\\'

#fileName = '6378LF,10m,100Krad_12-Nov-18_16-50_corrected'
#fileName = 'Coractive-ER8-6,7.5m,60 krad_05-Mar-19_19-24_Repaired'
#fileName = 'Coractive-SCF-ER60.2m,60 krad'    #No days given
fileName = 'pump2'

pastTime = 0


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

def get24Hour(Hour,Meridiem):
    # input: hour in 12h + am/pm
    # output: hour in 24h
    if Hour == 12 and Meridiem == 'AM':
        Hour = 0
    elif Hour == 12 and Meridiem == 'PM':
        Hour =  12
    elif Meridiem == 'PM':
        Hour = Hour + 12
    return Hour

def getTimes(df,pastTime):
    # This function reads an array of timestamps and returns an array of times in hours

    timeVec = np.zeros(len(df))
    
    timeVec[0] = pastTime
#    print(len(df))
    for i in range(0,len(df)):
#        print(i)
        
        data= df[i].split()
#        print(data)
        if data != []:
            Day, Month, Year = (data[0].split('/')) 
            Hour, Minute, Second = data[1].split(':')
            
            Day = int(Day)
            Month = int(Month)
            Year = int(Year)
            Hour = int(Hour)
            Minute = int(Minute)
            Second = float(Second)
        
        
            Meridiem = data[2]
            
            MonthDays = [0,31,59,89,120,150,181,212,243,273,304,334]
            
            time= (( MonthDays[int(Month)-1] + int(Day) )*24 + get24Hour(Hour,Meridiem))*3600 + Minute*60 +Second
            time = time/3600
            
            

            if i!=0:
                if timeVec[i-1] == 0 :
                    timeVec[i] = timeVec[i-2] + time-pastTime
                else:
                    timeVec[i] = timeVec[i-1] + time-pastTime
            
            pastTime = time
            
            
    return  timeVec  

#=============================================================================================

# This is the real function that uses the others     
def getTxtDataFrameWithTimes(folderPath,fileName,pastTime):
    #*********************************************************************************#
    #* This function reads a .txt with times in time stamps and returns hours        *#
    #* regradlles of time intervall between data point, but requires valid time stamp*#
    #*********************************************************************************#
    path = folderPath + fileName + '.txt'
     
    file = open(path)
    #print(file.readlines())

    df = file.readlines()
#    print(len(df))
    del df[0]

    timeVec = getTimes(df,pastTime)

    data = np.ones((len(df)-1,2))   
    for i in range(0,len(df)):
        tempData = df[i].split()
#        print(timeVec[i])
        if len(tempData) == 5:
                data[i-1,0] = timeVec[i]
                data[i-1,1] = tempData[3]


        elif len(tempData) == 4:
            print('get wrecked nub')
     

    for i in range(0,len(data)-10):
        if data[i,0] == 1:
            print(i)
            data = np.delete(data, (i), axis=0)
                 
    return data

data= getTxtDataFrameWithTimes(folderPath,fileName,pastTime)


