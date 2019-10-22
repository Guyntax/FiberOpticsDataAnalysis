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

#folderPath =  r'C:\Users\d.blach-lafleche\Desktop\photobleaching- files\\'
path =        [r'C:\Users\d.blach-lafleche\Desktop\Photobleaching\\',
               r'C:\Users\d.blach-lafleche\Documents\Python Scripts\Photobleaching_v3\30kRad\\',
               r'C:\Users\d.blach-lafleche\Documents\Python Scripts\Photobleaching_v3\60kRad\\',
               r'C:\Users\d.blach-lafleche\Documents\Python Scripts\Photobleaching_v3\100kRad\\',
               r'C:\Users\d.blach-lafleche\Documents\Python Scripts\Photobleaching_v3\6.5m\\',
               ]

#path, filename, G_0Rad [dBm], G_nRad [dBm],t_0 [hours], name in legend(graph) , getFit , PremadeDf, showFutur, offset
fileNameData = [[1,'6378LF,10m,30kRad_set-1+2+4_offset',19.68,18.38,0,'30 kRad',True,True,False,0], 
                [2,'6378LF,10m,60kRad_set1+3+5_offset',19.68,17.332,0,'60 kRad',True,True,False,0],#19.54,17.191
                [3,'6378Lf,10m,100kRad_set-1+3+5',19.68,16.139,0,'100 kRad set 1+2+3',True,True,False,0],
                ]

#path
#filename
#G_0Rad [dBm]
#G_nRad [dBm]
#t_0 [hours]
#name in legend(graph)
#getFit
#PremadeDf
#showFutur
#offset



#initialGuess = (0.09,0.20)

#initialGuess
LambdaVec = (0.003,)#(0.001 , 0.0015, 0.002 , 0.0025, 0.003 , 0.0035, 0.004)#np.arange(0.060,0.091,0.005) 
AlphaVec = (0.23,)#(0.18,0.19, 0.2 , 0.21, 0.22, 0.23, 0.24, 0.25, 0.26,) #np.arange(-0.200,-0.280,-0.01)




readData = True

saveGraphPNG = True        # Saves plots as .png 
showMaxPower = True # Show the maximum power value, which is power before irradiation  (G_n0Rad [dBm]) in graph
#showFutur = False    # Show futur values of the fitted function
futurMuliplier = 9.7 # Factor to multuply orginal dataset length
#setMaxValue = True  # Fit curve with max value parameter already set
getTimeData = True # Define time in dataframe by difference between points, instead of fixed interval
multiPlot = True
multiPlotName = 'No set'
combinePlots = False
unit = 'dBm'
#unit = 'mW'

# determine limits of graph, to zoom on a part
yLim = np.array([16]) #(17.6,18.6)
setXLim = False
xLim = (80,120)


exportDataFrame = True ## aucune idée ce que ca fait

fitPar = 'Alpha'
#
#Lambda = [0.67]
#AlphaVec = [-0.213,-0.233,-0.253,-0.273]
####=========================================================================================== 


Weibull  = lambda time,Lambda,Alpha,maxP,minP,: minP + (maxP-minP)*(1 - np.exp(-1*np.power((Lambda*(time-t0)),Alpha)))
Function = ['AccLoss function',Weibull ,2,[0.09,-0.23],[-300,-16]]

    ###############################################################################
def function(par,time,data,fit,findPar):
    
    Lambda= par[0]
    Alpha= par[1]
    
    FVU = np.empty(len(data))
    r2vec = np.empty(len(data))
    
#    print('momo')
    for i in range(0,len(data)):
#        print(i)
        fit[i] = f(time[i],Lambda,Alpha,maxPower[i],minPower[i])
        
        ssTot = np.sum((data[i]-np.average(data[i]))**2)
        ssRes = np.sum((data[i]-fit[i])**2)
        
        FVU[i] = (ssRes/ssTot) #fraction of variance unexplained
#        print(i)      
        if np.isnan(fit[i]).any() :
            r2vec[i] = 0
        else:
            r2vec[i] = r2_score(data[i],fit[i])
        
    if findPar:
        return 1 - np.average(r2vec)
    else:
        return r2vec
    ###############################################################################


# Script start here

# read text files and load data    
if readData:
    time = [None]*len(fileNameData)
    data = [None]*len(fileNameData)
    fit  = [None]*len(fileNameData)
    maxPower = [None]*len(fileNameData)
    minPower = [None]*len(fileNameData)

    radiation = [None]*len(fileNameData)
    getFit = [None]*len(fileNameData)
    
    for j in range(0, len(fileNameData)):
        print(fileNameData[j][0])
        
        # get all elemets from filenameData into independeant varibales with descriptive names
        folderPath = path[fileNameData[j][0]]
        fileName = fileNameData[j][1]
        maxPower[j] = fileNameData[j][2]
        minPower[j] = fileNameData[j][3]
        t0 = fileNameData[j][4]
        radiation[j] = fileNameData[j][5][0:8]
    #    radiation = fileNameData[j][4]
        getFit[j] = fileNameData[j][6]
        PremadeDf = fileNameData[j][7]
        showFutur = fileNameData[j][8]
        offset = fileNameData[j][9]
        
        # get data depinened on input parameter for configuration of text file
        if PremadeDf:
            df= getTxtPremadeDataFrame(folderPath,fileName)
        else:            
            if getTimeData:
                df= getTxtDataFrameWithTimes(folderPath,fileName,pastTime) 
            else:
                df= getTxtDataFrame(folderPath,fileName)
        if offset != 0:
            df = offsetData(df,offset)  
        
        time[j] = np.transpose(df)[0]
        data[j] = np.transpose(df)[1]


    f = Weibull
    

    

    





# loop to fit with different initial guesses
for i in LambdaVec:
    for k in AlphaVec:
        
        print('Alpha = {0} - Lambda = {1}'.format(i,k))
        initialGuess = (i,k)
        
        res = minimize(function,initialGuess,args=(time,data,fit,True),bounds=[(-2,10),(-2,10)])
        
        par = res['x']
        avgR2 =  res['fun']
        
        fit     = [None]*len(fileNameData)
        fitTime = [None]*len(fileNameData)
        
        R2Vec = function(par,time,data,fit,False)
        
        
        for m in range(0,len(data)):
            fitTime[m] = np.linspace(time[m][0],time[m][-1])
            fit[m] = f(fitTime[m],*par,maxPower[m],minPower[m])
        


        # make figure
        plt.figure()
        
        plt.plot(fitTime[0],maxPower[0]*np.ones(len(fitTime[0])))
        
        plt.scatter(time[0],data[0],s=0.1,c='red')
        plt.plot(fitTime[0],fit[0])
        
        plt.scatter(time[1],data[1],s=0.1,c='red')
        plt.plot(fitTime[1],fit[1])
        
        plt.scatter(time[2],data[2],s=0.1,c='red')
        plt.plot(fitTime[2],fit[2])
        
        
        plt.title('6378LF,10m - Weibull function - Lambda = {0:.3f}, Alpha = {1:.3f}'.format(*par))
        plt.legend(['0 kRad','30 kRad - r2 = {:.3f}'.format((R2Vec)[0]),'60 kRad - r2 = {:.3f}'.format((R2Vec)[1]),'100 kRad - r2 = {:.3f}'.format((R2Vec)[2])])
        plt.xlabel('Time [Hours]')
        plt.ylabel('Power [dBm]')
        
        title = '6378LF,10m - Weibull function - Initial Guess = {0} - Average R2 = {1:.4f}'.format(initialGuess,avgR2)
        
        
        # Save
        if saveGraphPNG:
            if os.path.exists(folderPath + 'AverageR2') == False:
                os.mkdir(folderPath + 'AverageR2')
            plt.savefig(folderPath + 'AverageR2' +'\\'+title + '.png', bbox_inches="tight")
    


