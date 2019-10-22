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

#folderPath =  r'C:\Users\d.blach-lafleche\Desktop\photobleaching- files\\'
path =  [r'C:\Users\d.blach-lafleche\Desktop\Photobleaching\\',
               r'C:\Users\d.blach-lafleche\Documents\Python Scripts\Photobleaching_v3\30kRad\\',
               r'C:\Users\d.blach-lafleche\Documents\Python Scripts\Photobleaching_v3\60kRad\\',
               r'C:\Users\d.blach-lafleche\Documents\Python Scripts\Photobleaching_v3\100kRad\\',
               r'C:\Users\d.blach-lafleche\Documents\Python Scripts\Photobleaching_v3\6.5m\\',
               ]

fileNameData = [
#                [1,'6378LF,10m,30krad_set-1',19.54,18.24,0,'30 kRad',True,True,True,0], #0.185


#                [2,'6378LF,10m,60kRad_set-1',19.54,17.191,0,'60 kRad',True,True,True,0], # 6 da
#                [2,'6378LF,10m,60kRad_set-2',19.54,17.9,141.09,'60 kRad',False,True,False,0],  
#                [2,'6378LF,10m,60kRad_set-3',19.54,17.191,0,'60 kRad',False,True,False,0], #0.135
#                [2,'6378LF,10m,60kRad_set-4',19.54,17.191,0,'60 kRad',False,True,False,0],
#                [2,'6378LF,10m,60kRad_set-5',19.54,17.191,0,'60 kRad',False,True,False,0],
#                [2,'6378LF,10m,60kRad_set1+3+4+5',19.54,17.191,0,'60 kRad',True,True,True,0],
                
                
#                [3,'6378Lf,10m,100kRad_set-1',19.54,16.039,0,'100 kRad set 1',False,True,False,False,0],
#                ['6378Lf,10m,100kRad_set-2',19.54,16.039,0,'100 kRad',False,True,False,0],              
#                ['6378Lf,10m,100kRad_set-3',19.54,16.039,0,'100 kRad',False,True,False,0],
#                ['6378Lf,10m,100kRad_set-4',19.54,16.039,0,'100 kRad',False,True,False,0],
                [3,'6378Lf,10m,100kRad_set-1+2+3+5',19.54,16.039,0,'100 kRad set 1+2+3',True,True,False,True,0], 
#                ['6378Lf,10m,100kRad_set-5',19.54,16.039,0,'100 kRad',False,True,False,0],
#                ['CombinedData',19.54,16.039,0,'100 kRad',True,True,False,0],

                ]

Weibull  = lambda time,a,b: minPower + (maxPower-minPower)*np.exp(-1*np.power((a*(time-t0)),b))
Function = ['AccLoss function',Weibull ,2,[0.09,-0.23],[-300,-16]]





save = False        # Saves plots as .png and r2 in .xls
showMaxPower = True # Show the maximum power value in graph
#showFutur = False    # Show futur values of the fitted function
futurMuliplier = 9.7 # Factor to multuply orginal dataset length
setMaxValue = True  # Fit curve with max value parameter already set
getTimeData = True # Define time in dataframe by difference between points, instead of fixed interval
multiPlot = True
multiPlotName = 'No set'
combinePlots = False
unit = 'dBm'
#unit = 'mW'
yLim = np.array([16]) #(17.6,18.6)
setXLim = False
xLim = (80,120)
exportDataFrame = True

fitPar = 'Alpha'
Alpha = [-0.244]
LambdaVec = [0.027,0.067,0.107,0.147] #
#Lambda = [0.67]
#AlphaVec = [-0.213,-0.233,-0.253,-0.273]
###=========================================================================================== 

if fitPar == 'Lambda':
    ParVec = AlphaVec
elif fitPar == 'Alpha':
    ParVec = LambdaVec

title = 'Weibull function'

plotData = [[],[],[],[],[],[],[]]
pMat = []
radiation = []

getFit = np.zeros(len(fileNameData), dtype=bool)


#print(fileNameData[0][0])
folderPath = path[fileNameData[0][0]]
fileName = fileNameData[0][1]
maxPower = fileNameData[0][2]
minPower = fileNameData[0][3]
t0 = fileNameData[0][4]
radiation = fileNameData[0][4]
getFit= fileNameData[0][6]
PremadeDf = fileNameData[0][7]
showFutur = fileNameData[0][8]
showData = fileNameData[0][9]
offset = fileNameData[0][10]

print(folderPath)
print(fileName)

if PremadeDf:
    df= getTxtPremadeDataFrame(folderPath,fileName)
else:            
    if getTimeData:
        df= getTxtDataFrameWithTimes(folderPath,fileName,pastTime) 
    else:
        df= getTxtDataFrame(folderPath,fileName)
if offset != 0:
    df = offsetData(df,offset)

time = np.array(df[:,0])
power = np.array(df[:,1])


# Generate plotData list
plotData[0] = time
plotData[1] = power

func = Function[1]
for i in range(0,len(ParVec)):
    if fitPar == 'Lambda':
        Lambda = Lambda
        Alpha = AlphaVec[i]
    elif fitPar == 'Alpha':
        Alpha =  Alpha
        Lambda = LambdaVec[i]
    
    plotData[2+i] = func(time,Lambda,Alpha)  



# Plot the data
#legendVec = ['max power']
#if showMaxPower:
#plt.plot(maxPowerTime, maxPower*np.ones(len(maxPowerTime)), label = 'maxPower')

for j in range(0,len(plotData)-1):
    if showData:
        plt.plot(plotData[0],plotData[1+j]) # Data
#        legendVec.append('{}-power'.format(1))
        
#    if getFit[j]:
#        for k in range(0,len(ParVec)):
#            plt.plot(plotData[j][i][k+1][0],plotData[j][i][k+1][1]) # Fit   
#            
#            legendVec.append('{0}-fit, {2} = {1}'.format(k,ParVec[k],fitPar))                 
##                parBox(ax,[ParVec[k]],.92,0.5-(0.15*(j+k)),k)
#        
#bottom, top = plt.ylim()  # return the current ylim
##    
#if len(yLim) == 2:
#    plt.ylim(yLim)  
#else:
#    plt.ylim((yLim, top)) 
#if setXLim:
#    plt.xlim(xLim) 
#plt.title(title)
#plt.legend(legendVec, loc = 'upper left',bbox_to_anchor=(1, 1.02))
#plt.xlabel('Time [Hours]')
#plt.ylabel('Power [{}]'.format(unit))


#    yticksLabels = [17,'','','','',17.5,'','','','',18,'','','','',18.5,'','','','',19,'','','','',19.5]
#    plt.yticks(np.arange(17, 19.6, step=0.1),yticksLabels)
plt.grid(linestyle='--', linewidth=0.5)