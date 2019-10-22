import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf
from sklearn.metrics import r2_score
import os

from getTxtDataFrame_v3 import getTimeInterval
from getTxtDataFrame_v3  import getTxtDataFrame
from getTxtDataFrame_v3 import getTxtPremadeDataFrame
from getTxtDataFrame_v3 import offsetData
from getTxtDataFrameWithTimes_v3  import getTxtDataFrameWithTimes
from getTxtDataFrameWithTimes_v3  import getTimes
from getTxtDataFrameWithTimes_v3  import get24Hour
###===========================================================================================



#folderPath =  r'C:\Users\d.blach-lafleche\Desktop\photobleaching- files\\'
folderPath =  r'C:\Users\d.blach-lafleche\Desktop\Photobleaching\\'
folderPath =  r'C:\Users\d.blach-lafleche\Documents\Python Scripts\Photobleaching_v3\\'

fileNameData = [#['2262LF,8m,60 krad_23-Nov-18_17-12',13.424], #6 days
                #['2262LF,8m,100Krad_04-Dec-18_18-28',20.32], #overflow in multiply for capacitor, not a problem # 9 days
                ######'4243LF,15m,30Krad_26-Sep-18_14-12', # no Max Power data # 8 days
                
                #['6387LF,10m,10Krad_Repaired',19.54,19.202,0,'10 kRad',True,False],     #No days given
                #['6387LF,10m,30Krad__01-Nov-18_17-44',19.54,18.24,0,'30 kRad',True,False],  #% 5 days 
                #['6378LF,10m,30krad_J0B3_14-25-26-27 Juin',19.54,18.24,120.059,'30 kRad',False,False],
                
                
                ['6387LF,10m,60Krad__06-Nov-18_18-42',19.54,17.191,0,'60 kRad',True,False], # 6 days 
                #['6387LF,10m,60Krad__set1+2+3_bon_ajusted',19.54,17.191,141.075,'60 kRad',False,True], # 6 days ]
                ['6378LF,10m,30krad_J0B3_set1',19.54,18.24,0,'30 kRad',True,True,True,0],  #% 5 days 
                #['6378LF,10m,30krad_J0B3_set2.1',19.54,18.24,120.059,'30 kRad',False,True,False],
#                ['6378LF,10m,30krad_J0B3_set2',19.54,18.24,120.059,'30 kRad',False,True,False,0.185],
                #['6378LF,10m,30krad_J0B3_set1+2',19.54,18.24,120.059,'30 kRad',True,True],
                
#                ['6378LF,10m,60Krad_J0B4_set1',19.54,17.191,0,'60 kRad',True,True,True,0], # 6 da
#                ['6378LF,10m,60Krad_J0B4_set2.1',19.54,17.191,141.075,'60 kRad',False,True,False,0.537],
#                ['6378LF,10m,60Krad_J0B4_set2',19.54,17.191,141.075,'60 kRad',False,True,False,0], #0135
                #['6378LF,10m,60Krad_J0B4_set1+2',19.54,17.191,0,'60 kRad',True,True],
                
                

                ['6378Lf,10m,100kRad_set-1',19.54,16.039,0,'100 kRad',True,True,True,0],
#                ['6378Lf,10m,100kRad_set-2',19.54,16.039,0,'100 kRad',False,True,False,0],              
#                ['6378Lf,10m,100kRad_set-3',19.54,16.039,0,'100 kRad',False,True,False,0],
#                ['6378Lf,10m,100kRad_set-4',19.54,16.039,0,'100 kRad',False,True,False,0],
#                ['6378Lf,10m,100kRad_set-1+2+3',19.54,16.039,0,'100 kRad',True,True,True,0], 
#                ['6378Lf,10m,100kRad_set-5',19.54,16.039,0,'100 kRad',False,True,False,0],
#                ['CombinedData',19.54,16.039,0,'100 kRad',True,True,False,0],

                
                #['6378LF,10m,100Krad_12-Nov-18_16-50',19.54,16.039,0,'100 kRad',True,False], # 4 days
                #['6378LF,10m,100krad_set1',19.54,16.039,0,'100 kRad',True,True], # 4 days
                #['6378LF,10m,100krad_set2',19.54,16.039,96.5701,'100 kRad',False,True], #
                #['6378LF,10m,100krad_set1+2',19.54,16.039,0,'100 kRad',True,True],     
                #['6378LF,10m,100krad_set3',19.54,16.039,96.5701,'100 kRad',False,True],
       ##         ['6378LF,10m,100krad,set1+2+3',19.54,16.039,0,'100 kRad',True,True,False],


#                ['6378Lf,6.5m,100kRad_set-0',19.69,17.25,0,'6.5m',True,True,True,0],
#                #['6378Lf,6.5m,100kRad_set-1',19.69,17.25,0,'6.5m',False,True,False,0],
#                ['6378Lf,6.5m,100kRad_set-2',19.69,17.25,0,'6.5m',False,True,False,-0.03],
#                #['6378Lf,6.5m,100kRad_set-3',19.69,17.25,0,'6.5m',False,True,False,0.7+0.05],
#                ['6378Lf,6.5m,100kRad_set-4',19.69,17.25,0,'6.5m',False,True,False,0],
#                #['6378Lf,6.5m,100kRad_set-5',19.69,17.25,0,'6.5m',False,True,False,0.7+0.05],
#                ['6378Lf,6.5m,100kRad_set-6',19.69,17.25,0,'6.5m',False,True,False,0],
#                #['6378Lf,6.5m,100kRad_setFinal',19.69,17.25,0,'6.5m',False,True,False,0]

                #['6378Lf,6.5m,100kRad_set1_cut',19.69,17.25,0,'6.5m',True,True,True,0],
                #['6378Lf,6.5m,100kRad_set2_cut',19.69,17.25,0,'6.5m',False,True,False,0],
                #['6378Lf,6.5m,100kRad_set1+2_cut',19.69,17.25,0,'6.5m',True,True,False,0],


                #['Coractive-ER-12,10m,30 Krad_01-Mar-19_20-21',19.8], # 1 day
                #['ER8-6,7.5m,30 Krad_17-Oct-18_16-57',18.97], # 1 day 
                #['ER8-6,7.5m,30Krad_1_18-Oct-18_17-56',18.97], #11 days    
                #['coractive-ER60,2m,10krad_28-Feb-19_18-53',19], #1 day
                #['Coractive-SCF-ER-60,2m,30krad',19], #No days given
                #['Coractive-SCF-ER60.2m,60 krad',19], #No days given
                #['ER60_2m,100 krad_19-Dec-18_17-40_Repaired',19], # 2 days 
                ]

save = False        # Saves plots as .png and r2 in .xls
showMaxPower = True # Show the maximum power value in graph
#showFutur = False    # Show futur values of the fitted function
futurMuliplier = 1.8 # Factor to multuply orginal dataset length
setMaxValue = True  # Fit curve with max value parameter already set
getTimeData = True # Define time in dataframe by difference between points, instead of fixed interval
multiPlot = True
multiPlotName = 'No set'
combinePlots = False
unit = 'dBm'
#unit = 'mW'
yLim = 16
setXLim = False
xLim = (900,1000)
ExportDataFrame = True


###=========================================================================================== 

#define function for fitting

if setMaxValue:
    ArcTanFunc = lambda time,a,b : (maxPower*2/np.pi)*np.arctan(a*(time - b))
    InvFunc = lambda time,a,b: a/(time-b) + maxPower
    Logfunc = lambda time,a,b: a*np.log(time) +b
    ExpsumFunc = lambda time,a,b,c,d: maxPower + np.exp(a*(time-b))-np.exp(c*(time-d))
    AccLossFunc = lambda time,a,b: minPower + (maxPower-minPower)*np.exp(-1*np.power((a*time),b))
    AccLossFunc_s = lambda time,b: minPower + (maxPower-minPower)*np.exp(-1*np.power((0.089*time),b))#0.67-0.89
    AccLossFunc_lambda = lambda time,a: minPower + (maxPower-minPower)*np.exp(-1*np.power((a*time),-0.245))#-0.215 - -0.244
    
    Function = [#['Log function', Logfunc,2,[1,1],'dBm'],
                #['ArcTan function', ArcTanFunc,3,[2,3],'dBm'],
                #['Inverse function', InvFunc,3,[-2,-5],[-300,-16]],
                
                ['AccLoss function',AccLossFunc,2,[0.166,1],[-300,-16]],  #(0.09,-0.23)
                #['AccLossFunc_s',AccLossFunc_s,1,[0.137],[-300,-16]], 
               # ['AccLossFunc_lambda',AccLossFunc_lambda,1,[0.08],[-300,-16]],
                ]
else:
    ArcTanFunc = lambda time,a,b,c : a*np.arctan(b*(time - c))
    InvFunc = lambda time,a,b,c: a/(time-b) + c
    ExpsumFunc = lambda time,a,b,c,d,e: e + np.exp(a*(time-b))-np.exp(c*(time-d))
    
    
    Function = [#['Log function', Logfunc,2,[1,1],[-2,-5]],
                #['ArcTan function', ArcTanFunc,3,[2,3,3],[-2,-5]],
                #['Inverse function', InvFunc,3,[-2,-5,18],[-300,-15]],
                #['ExpSum function', ExpsumFunc,2,[10,-1,10,-1,19],[1,1]]
                ] 
    
###===========================================================================================   
    
# Function used in plots to generate parameter boxes    
def parBox(ax,p,x,y):
    parName = ['lambda','alpha','c','d','e']
    string = ''
    for i in range(0,len(p)):
        string = string + parName[i] + '={'+'{}:.3f'.format(i)+'} '
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)

    ax.text(x,y, 'Parameters : \n '+string.format(*p), fontsize=12,
            verticalalignment='top', bbox=props)       
    
###===========================================================================================

# Function that plots the figures in all the different required ways
def plotFunc(ax,title,time,power,maxPower,fitTime,fit,r2,
                 folderPath,fileName,showMaxPower,showFutur,unit ):
 

    plt.title(title)
    plt.plot(time, power, label = 'power')
    if getFit[j]:
        plt.plot(fitTime, fit, label  = 'fit')
    if showMaxPower:
        plt.plot(fitTime, maxPower*np.ones(len(fitTime)), label = 'maxPower')
    plt.legend(('power','fit : r2 = {0:.4f}'.format(r2),'max power'),
               loc = 'lower right',bbox_to_anchor=(1.4, -0.02))
    
    bottom, top = plt.ylim()  # return the current ylim
    plt.ylim((yLim, top)) 

    plt.xlabel('Time [Hours]')
    plt.ylabel('Power [{}]'.format(unit))
    
    parBox(ax,p,0.925,0.5)
    
        
###===========================================================================================   
    
# Function that multiplots the figures in all the different required ways
def multiPlotfunc(ax,i,plotData,title,r2Mat,showMaxPower,maxPowerTime,getFit):
    # i is function index
    # j is data set index

    legendVec = ['max power']
    if showMaxPower:
        plt.plot(maxPowerTime, maxPower*np.ones(len(maxPowerTime)), label = 'maxPower')

    for j in range(0,len(plotData)):
        plt.plot(plotData[j][i][0][0],plotData[j][i][0][1]) # Data
        if getFit[j]:
            plt.plot(plotData[j][i][1][0],plotData[j][i][1][1]) # Fit  
            if radiation == []:
                legendVec.append('{}-power'.format(j+1))
                legendVec.append('{0}-fit : r2 = {1:.4f}'.format(j+1,r2Mat[j][i]))
            else:
                legendVec.append('{}-power'.format(radiation[j]))
                legendVec.append('{0}-fit : r2 = {1:.4f}'.format(radiation[j],r2Mat[j][i]))
    #        print(pMat[j][i])
            parBox(ax,np.array(pMat[j][i]),1.32,0.86-(0.15*j))

    bottom, top = plt.ylim()  # return the current ylim
    plt.ylim((yLim, top)) 
    if setXLim:
        plt.xlim(xLim) 
    plt.title(title)
    plt.legend(legendVec, loc = 'upper left',bbox_to_anchor=(1, 1.02))
    plt.xlabel('Time [Hours]')
    plt.ylabel('Power [{}]'.format(unit))
    
    
###===========================================================================================


#  WTF does this do !?!??     
titleVec = np.empty(len(Function), dtype = 'U50')
r2Vec = np.empty(len(Function))

for i in range(0,len(Function)):
    
    titleVec[i]  = Function[i][0]
    #print(Function[i][0])
r2data = {'Functions' : titleVec}            




 # main loop start here   
plotData = []
pMat = []
#radiation = []
radiation = [['            '],['            '],['            '],['            '],['            '],['            '],['            '],['            ']]
getFit = np.zeros(len(fileNameData), dtype=bool)
r2Mat = np.ones([len(fileNameData),len(Function)])
#r2Mat = []       
for j in range(0, len(fileNameData)):
    print(fileNameData[j][0])
    
    fileName = fileNameData[j][0]
    maxPower = fileNameData[j][1]
    minPower = fileNameData[j][2]
    pastTime = fileNameData[j][3]
    radiation[j] = fileNameData[j][4][0:8]
#    radiation = fileNameData[j][4]
    getFit[j] = fileNameData[j][5]
    PremadeDf = fileNameData[j][6]
    showFutur = fileNameData[j][7]
    offset = fileNameData[j][8]
    
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
    
        # convert to  dbm or mW
#        unit = Function[i][4]
    if unit == 'mW':
#        powerMW = np.power(10,power/10)
#        maxPowerMW = np.power(10,maxPower/10)
        power = np.power(10,power/10)
        maxPower = np.power(10,maxPower/10)
        
    # Find, plot and save best fit
    pVec = []
    plotDataTemp = []  
    for i in range(0,len(Function)):
        
        title  = Function[i][0]
        func = Function[i][1]
        numPar = Function[i][2]
        if unit == 'dBm': 
            p0  =  Function[i][3]   
        else:
            p0  =  Function[i][4] 
    
        # find optimal parameters
        print(title)
#        if unit == 'mW':
#            p,cov = curve_fit(func, time, powerMW, p0, maxfev=2000)
#        else:
#            p,cov = curve_fit(func, time, power, p0, maxfev=2000)
#        time = np.array(time, dtype=np.complex)
        p,cov = curve_fit(func, time, power, p0, maxfev=8000)
        
        
        # compute fit
        fit = func(time, *p)
#        if unit == 'mW':
#            r2Vec[i] = r2_score(powerMW,fit)
#        else:
#            r2Vec[i] = r2_score(power,fit)
        r2Vec[i] = r2_score(power,fit)
            
        # futur fit
        if showFutur:
            futurTime = np.linspace(max(time)+0.1,futurMuliplier*max(time),200000)
            futurFit = func(futurTime, *p)
            fitTime = np.concatenate([time,futurTime])
            fit = np.concatenate([fit,futurFit])
        else:
            fitTime = time
            fit = fit
        if j == 0 :
            maxPowerTime = fitTime
        # Calls the plot function or prepares the multiplot data  
        if multiPlot:  
            tempList = [np.array([time,power]),np.array([fitTime,fit])]
            plotDataTemp.append(tempList)
#            print(r2Vec)
            
        else:
            ax = plt.figure()
            if unit == 'mW':
                plotFunc(ax,title,time,powerMW,maxPowerMW,fitTime,fit,r2Vec[i],
                         folderPath,fileName,showMaxPower,showFutur,unit)       
            else:
                plotFunc(ax,title,time,power,maxPower,fitTime,fit,r2Vec[i],
                         folderPath,fileName,showMaxPower,showFutur,unit)
    
            if save:
                if os.path.exists(folderPath + fileName) == False:
                    os.mkdir(folderPath + fileName )
                plt.savefig(folderPath + fileName +'\\'+ unit+title + '.png', bbox_inches="tight")
#            plt.close(ax)
        pVec.append(p.tolist()[:])  
            
#    r2Mat.append(r2Vec[:])
    pMat.append(pVec[:])      
    r2Mat[j] = r2Vec
    plotData.append(plotDataTemp)    
print(len(plotData))
if multiPlot:
    for i  in range(0,len(plotData[0])):
#        r2Mat = np.array([[1,3],[5,2],[5,1],[3,4]])
        title  = Function[i][0]
        ax = plt.figure()
#        if unit == 'mW':
#            plotFunc(ax,title,time,powerMW,maxPowerMW,fitTime,fit,r2Vec[i],
#                         folderPath,fileName,showMaxPower,showFutur,unit)       
#        else:
#            plotFunc(ax,title,time,power,maxPower,fitTime,fit,r2Vec[i],
#                         folderPath,fileName,showMaxPower,showFutur,unit)        
        multiPlotfunc(ax,i,plotData,title,r2Mat,showMaxPower,maxPowerTime,getFit)
        if save:
            if os.path.exists(folderPath + 'multiPlot') == False:
                os.mkdir(title )
            plt.savefig(folderPath + 'multiPlot' +'\\'+ unit+title + '.png', bbox_inches="tight")


# =============================================================================
#     print(r2Vec)
#     r2data.update({fileName: r2Vec})
#     print(r2data)
#     
# fileNameVec = np.empty(len(fileNameData), dtype = 'U50')
# for i in range(0,len(fileNameData)):
#     fileNameVec[i]='set'+str(i+1)
# =============================================================================
    
r2dataFrame = pd.DataFrame(r2data) # , columns= ['Functions', *fileNameData])
#print(r2dataFrame)
if save:
    export_excel = r2dataFrame.to_excel (folderPath + r'dataframe.xlsx', index = None, header=True)
###########################################################



