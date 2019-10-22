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

fitPar = 'Lambda'
LambdaVec = [0.67]# [0.27,0.67,1.07,1.57]
AlphaVec = [-0.213,-0.233,-0.253,-0.273]
###=========================================================================================== 

#define function for fitting

if setMaxValue:
    ArcTanFunc = lambda time,a,b : (maxPower*2/np.pi)*np.arctan(a*(time - b))
    InvFunc = lambda time,a,b: a/(time-b) + maxPower
    Logfunc = lambda time,a,b: a*np.log(time) +b
    ExpsumFunc = lambda time,a,b,c,d: maxPower + np.exp(a*(time-b))-np.exp(c*(time-d))
    AccLossFunc = lambda time,a,b: minPower + (maxPower-minPower)*np.exp(-1*np.power((a*(time-t0)),b))
    AccLossFunc_s = lambda time,b: minPower + (maxPower-minPower)*np.exp(-1*np.power((Lambda*(time-t0)),b))#0.67-0.89
    AccLossFunc_lambda = lambda time,a: minPower + (maxPower-minPower)*np.exp(-1*np.power((a*(time-t0)),Alpha))#-0.215 - -0.244
    
    Function = [#['Log function', Logfunc,2,[1,1],'dBm'],
                #['ArcTan function', ArcTanFunc,3,[2,3],'dBm'],
                #['Inverse function', InvFunc,3,[-2,-5],[-300,-16]],
                
                #['AccLoss function',AccLossFunc,2,[0.09,-0.23],[-300,-16]],  #(0.09,-0.23)   6.5 = 0.166,1
                ['AccLossFunc_s',AccLossFunc_s,1,[-1],[-300,-16]], 
                #['AccLossFunc_lambda',AccLossFunc_lambda,1,[0.09],[-300,-16]],
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
def parBox(ax,p,x,y,k):
    print('p = '+ str(p))
#    parName = ['lambda','alpha','c','d','e']
    parName = ['Alpha']
    string = ''
    for i in range(0,len(p)):
        string = string + parName[i] + '={'+'{}:.3f'.format(i)+'} '
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)

    ax.text(x,y, '{}- Parameters : \n '.format(k)+string.format(*p), fontsize=12,
            verticalalignment='top', bbox=props)       
    
###==========================================================================================

# Function that plots the figures in all the different required ways
def plotFunc(ax,title,time,power,maxPower,fitTime,fit,r2,
                 folderPath,fileName,showMaxPower,showFutur,unit):
 

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
        if showData:
            plt.plot(plotData[j][i][0][0],plotData[j][i][0][1]) # Data
            legendVec.append('{}-power'.format(1))
            
        if getFit[j]:
            for k in range(0,len(ParVec)):
                plt.plot(plotData[j][i][k+1][0],plotData[j][i][k+1][1]) # Fit   
                
                legendVec.append('{0}-fit, {2} = {1}'.format(k,ParVec[k],fitPar))                 
                parBox(ax,[(pMat[j][i][k])],.92,0.5-(0.15*(j+k)),k)
            
    bottom, top = plt.ylim()  # return the current ylim
#    
    if len(yLim) == 2:
        plt.ylim(yLim)  
    else:
        plt.ylim((yLim, top)) 
    if setXLim:
        plt.xlim(xLim) 
    plt.title(title)
    plt.legend(legendVec, loc = 'upper left',bbox_to_anchor=(1, 1.02))
    plt.xlabel('Time [Hours]')
    plt.ylabel('Power [{}]'.format(unit))


#    yticksLabels = [17,'','','','',17.5,'','','','',18,'','','','',18.5,'','','','',19,'','','','',19.5]
#    plt.yticks(np.arange(17, 19.6, step=0.1),yticksLabels)
#    plt.grid(linestyle='--', linewidth=0.5)
    
def getDataFrame(plotData):
    for i in range(0,len(plotData)):
        for j in range(0,len(plotData[i])):
            for k in range(0,len(plotData[i][j])):
                file = open(folderPath + "data"+str(i)+str(j)+str(k)+".txt","w")
                dataset = np.transpose(plotData[i][j][k])
                for l  in range(0,len(dataset)):
                    file.write("{} ".format(str(dataset[l][0])) + "{} ".format(str(dataset[l][1])) + "\n") 
                file.close() 
    
###===========================================================================================

if fitPar == 'Lambda':
    ParVec = LambdaVec
elif fitPar == 'Alpha':
    ParVec = AlphaVec

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
radiation = []
#radiation = [['            '],['            '],['            '],['            '],['            '],['            '],['            '],['            ']]
getFit = np.zeros(len(fileNameData), dtype=bool)

r2Mat = np.ones([len(fileNameData),len(Function),len(ParVec)])
#r2Mat = []       
for j in range(0, len(fileNameData)):
    print(fileNameData[j][0])
    folderPath = path[fileNameData[j][0]]
    fileName = fileNameData[j][1]
    maxPower = fileNameData[j][2]
    minPower = fileNameData[j][3]
    t0 = fileNameData[j][4]
#    radiation[j] = fileNameData[j][5][0:8]
#    radiation = fileNameData[j][4]
    getFit[j] = fileNameData[j][6]
    PremadeDf = fileNameData[j][7]
    showFutur = fileNameData[j][8]
    showData = fileNameData[j][9]
    offset = fileNameData[j][10]
    
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
        

        fit = [0] * len(ParVec)
        r2Vec = np.empty((len(Function),len(ParVec)))
        pArray = np.empty(len(ParVec))
        for k in range(0,len(ParVec)):
            if fitPar == 'Lambda':
                Lambda = ParVec[k]
            elif fitPar == 'Alpha':
                Alpha = ParVec[k]


            p,cov = curve_fit(func, time, power, p0, maxfev=8000)
            pArray[k] = p
    
        # compute fit
            fit[k]= func(time, *p)
#        if unit == 'mW':
#            r2Vec[i] = r2_score(powerMW,fit)
#        else:
#            r2Vec[i] = r2_score(power,fit)
            r2Vec[i][k]= r2_score(power,fit[k])
            
        # futur fit
        if showFutur:
            futurTime = np.linspace(max(time)+0.1,futurMuliplier*max(time),200000)
            futurFit = func(futurTime, *p)
            fitTime = np.concatenate([time,futurTime])
            fit[k] = np.concatenate([fit[k],futurFit])
        else:
            fitTime = time
            fit[k] = fit[k]
        if j == 0 :
            maxPowerTime = fitTime
        # Calls the plot function or prepares the multiplot data  
        if multiPlot:
            fitArray = []
            for k in range(0,len(ParVec)):
                fitArray.append(np.array([fitTime,fit[k]]))
            tempList = [np.array([time,power]),*fitArray]
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
##################################################################################################################################  
        pVec.append(pArray.tolist()[:])  
             
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
#                os.mkdir(title )
                os.mkdir(folderPath + 'multiPlot')
            plt.savefig(folderPath + 'multiPlot' +'\\'+ unit+title + '.png', bbox_inches="tight")
#            plt.savefig(folderPath + 'multiPlot' +'\\'+'Parameter\\'+ unit+title+'_'+str(Alpha) + '.png', bbox_inches="tight")


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

#getDataFrame(plotData)

