#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize
#from scipy.special import erf
from sklearn.metrics import r2_score
#from getTxtDataFrame import getTimeInterval
#from astropy.table import Table
#import os

#from getTxtDataFrame_v3 import getTimeInterval
#from getTxtDataFrame_v3  import getTxtDataFrame
#from getTxtDataFrame_v3 import getTxtPremadeDataFrame
#from getTxtDataFrameWithTimes_v3  import getTxtDataFrameWithTimes
#from getTxtDataFrameWithTimes_v3  import getTimes
#from getTxtDataFrameWithTimes_v3  import get24Hour



folderPath = r'C:\Users\d.blach-lafleche\Desktop\Additional analysis\\'
fileName = 'second batch'


#save = False   


p0 = (1,1)
#maxPower =19.49
#minPower =17.27

Weibull  = lambda time,Lambda,Alpha: minPower + (maxPower-minPower)*(1 - np.exp(-1*np.power((Lambda*(time)),Alpha)))







file = open(folderPath + fileName + '.txt',"r")
###############################################
data = file.readlines()
###############################################
file.close()

#time = np.ones(len(df))
#power = np.ones(len(df))

plotData = [None]*len(data)
for j in range(0,len(data)):

    data[j] = (data[j].split())
    
    name     = data[j][0]
    company  = data[j][1]
    length   = data[j][2]
    jig      = data[j][3]
    pump     = data[j][4]
    maxPower = float(data[j][5])
    minPower = float(data[j][6])
    data1    = float(data[j][7])
    data2    = float(data[j][8])
    
    time  = [0,8,12]
    power = [minPower,data1,data2]
    



    p,cov = curve_fit(Weibull, time, power, p0, maxfev=50000)
    
    fit = np.empty(len(time))
    for i in range(0, len(time)):
        fit[i] = Weibull(time[i],*p)
    
    r2 = r2_score(power,fit)
    
    maxPowerVec = maxPower*np.ones(len(power))
    fitTime = np.linspace(time[0],time[-1],25)
    
    fit = np.empty(len(fitTime))
    for i in range(0, len(fitTime)):
        fit[i] = Weibull(fitTime[i],*p)
       
    
    plotData[j] = [time,power,fitTime,fit,p]
#    
#    plt.plot(time,maxPowerVec)
#    plt.scatter(time,power)
#    plt.plot(fitTime,fit)
#    plt.title(title + ' - '+ length +' - ' + pump +' - Lambda= {0:.6f} Alpha = {1:.6f}'.format(*p))
#    plt.xlabel('Time [Hours]')
#    plt.ylabel('Power [dBm]')
#    plt.legend(['max power','data','fit - r2 = {:.4f}'.format(r2)])
    
#    if save:
#        if os.path.exists(folderPath + 'Figures') == False:
#            os.mkdir(folderPath + 'Figures')
#        plt.savefig(folderPath + 'Figures' +'\\'+fileName + '.png', bbox_inches="tight")
#        
#        plt.close()


