#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize
#from scipy.special import erf
from sklearn.metrics import r2_score
#from getTxtDataFrame import getTimeInterval
#from astropy.table import Table
import os

from getTxtDataFrame_v3 import getTimeInterval
from getTxtDataFrame_v3  import getTxtDataFrame
from getTxtDataFrame_v3 import getTxtPremadeDataFrame
from getTxtDataFrameWithTimes_v3  import getTxtDataFrameWithTimes
from getTxtDataFrameWithTimes_v3  import getTimes
from getTxtDataFrameWithTimes_v3  import get24Hour



folderPath = r'C:\Users\d.blach-lafleche\Desktop\Additional analysis\\'
fileNameData = [#'GP980,OFS,MPB,Er,11m,IP,J0_B1',
                #'GP980 ,IP,contra-pumped',
                'GP980 , OFS,  Lot GP8A4301, IP,70C',
#                'GP980 ,IP,11days at 70C',
#                'GP980 ,INP,heated 8 days at70C',
#                'GP980 loaded with D2,IP,13days at 70C', # Max Power is wrong
                #####'GP980 loaded with D2,INP', #no pumping
#                'AG980H (high rad), ( Fibercore),IP,heated 13 days',
#                'AG980H, Fibercore,INP,pumped 2 weeks ,70C 8days', #min oower 0 au lieu de -0.8
#                'AG980L Fibercore,INP,8days in oven at 70C 8 days',
#                'ER30  Thorlabs,IP', #min oower 0 au lieu de -3.32
#                'Er PM,INP,annealed 12days at 70 C',
#                'Er PM,IP,17m',
#                'Er PM,IP,8m',
#                'DHB1500 (FiberCoreErDF ),INP',
                
                ]


save = False

Weibull  = lambda time,Lambda,Alpha: minPower + (maxPower-minPower)*(1 - np.exp(-1*np.power((Lambda*(time)),Alpha)))


p0 = (0.02,0.2)
maxPower =19.49
minPower =17.27

for i in range(0,len(fileNameData)):
    fileName = fileNameData[i]
    
    file = open(folderPath + fileName + '.txt',"r")
    
    title = file.readline()
    length = file.readline()
    pump =  file.readline()
    maxPower = float(file.readline())
    minPower = float(file.readline())
    file.readline()
    
    df = file.readlines()
    file.close()
    
    time = np.ones(len(df))
    power = np.ones(len(df))
    
    for j in range(0,len(df)):

        df[j] = (df[j].split())
        time[j] = float(df[j][0])
        power[j]= float(df[j][1])
        

    
    
    p,cov = curve_fit(Weibull, time, power, p0, maxfev=8000)
    
    
    
    print(p)
    
    fit = np.empty(len(time))
    for i in range(0, len(time)):
        fit[i] = Weibull(time[i],*p)
    
    r2 = r2_score(power,fit)
    
    maxPowerVec = maxPower*np.ones(len(power))
    fitTime = np.linspace(time[0],time[-1],50000)
    
    fit = np.empty(len(fitTime))
    for i in range(0, len(fitTime)):
        fit[i] = Weibull(fitTime[i],*p)
       
    
    plotData = [time,power,fitTime,fit]
    
    plt.plot(time,maxPowerVec)
    plt.scatter(time,power)
    plt.plot(fitTime,fit)
    plt.title(title + ' - '+ length +' - ' + pump +' - Lambda= {0:.6f} Alpha = {1:.6f}'.format(*p))
    plt.xlabel('Time [Hours]')
    plt.ylabel('Power [dBm]')
    plt.legend(['max power','data','fit - r2 = {:.4f}'.format(r2)])
    
    if save:
        if os.path.exists(folderPath + 'Figures') == False:
            os.mkdir(folderPath + 'Figures')
        plt.savefig(folderPath + 'Figures' +'\\'+fileName + '.png', bbox_inches="tight")
        
        plt.close()


