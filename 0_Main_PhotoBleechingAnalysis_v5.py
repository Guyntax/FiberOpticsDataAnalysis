import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize
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
###============================================================================

#*****************************************************************************#
#* Title: PhotoBleaching Analysis                                            *#
#* Author: Didier Blach-Lalfèche                                             *#
#* Date: August 16th 2019                                                    *#
#* Company: MPBC                                                             *#
#* Description: This script is for analysing powermeter data from optical    *#
#*              amplifiers being pumped after being irradiated. This script  *#
#*              will fit different non-linear function to the data. The best *#
#*              model is currently the Weibull function                      *#
#*                                                                           *#
#* Other script nessecary: getTxtDataFrame_v3 (Contains functions)           *#
#*                         getTxtDataFrameWithTimes_v3 (Contains functions)  *#
#*                         Data_merging_v2 (Used to prepare data)            *#
#*****************************************************************************#


# MainScriptFolderPath = 'C:\Users\d.blach-lafleche\Documents\Python Scripts\'
# ==> MainScriptFolderPath = 'Didiers_Photobleaching_Analyis_Summer2019\Data'

path =  [r'C:\Users\d.blach-lafleche\Desktop\Photobleaching\\',
               r'C:\Users\d.blach-lafleche\Documents\Python Scripts\Photobleaching_v3\30kRad\\',
               r'C:\Users\d.blach-lafleche\Documents\Python Scripts\Photobleaching_v3\60kRad\\',
               r'C:\Users\d.blach-lafleche\Documents\Python Scripts\Photobleaching_v3\100kRad\\',
               r'C:\Users\d.blach-lafleche\Documents\Python Scripts\Photobleaching_v3\6.5m\\',
               ]

fileNameData = [#[1,'6378LF,10m,30kRad_set-1+3',19.54,18.24,0,'30 kRad',True,True,False,0], 
                #[1,'6378LF,10m,30kRad_set-1+4',19.54,18.24,0,'30 kRad',True,True,False,0], 
                #[1,'6378LF,10m,30kRad_set-1+2+3',19.54,18.24,0,'30 kRad',True,True,False,0], 
                #[1,'6378LF,10m,30kRad_set-1+2+4',19.54,18.24,0,'30 kRad',True,True,False,0], 
                #[1,'6378LF,10m,30kRad_set-1+2+4_offset',19.68,18.38,0,'30 kRad',True,True,False,0], 
#                [1,'6378LF,10m,30kRad_set-1',19.68,18.38,0,'30 kRad',True,True,True,0], #0.185  # 19.54,18.24
#                [1,'6378LF,10m,30kRad_set-2',19.68,18.38,0,'30 kRad',False,True,False,0], 
#                [1,'6378LF,10m,30kRad_set-3',19.68,18.38,0,'30 kRad',False,True,False,0.03], 
#                [1,'6378LF,10m,30kRad_set-4',19.68,18.38,0,'30 kRad',False,True,False,0], 
                
                
                #[2,'6378LF,10m,60kRad_set-1',19.68,17.332,0,'60 kRad',True,True,True,0], # 6 da
#                [2,'6378LF,10m,60kRad_set-2',19.54,17.191,141.09,'60 kRad',False,True,False,0],  
                #[2,'6378LF,10m,60kRad_set-3',19.54,17.191,0,'60 kRad',False,True,False,0], #0.135
#                [2,'6378LF,10m,60kRad_set-4',19.54,17.191,0,'60 kRad',False,True,False,0],
                #[2,'6378LF,10m,60kRad_set-5',19.54,17.191,0,'60 kRad',False,True,False,0],
#                [2,'6378LF,10m,60kRad_set1+3+4+5',19.54,17.191,0,'60 kRad',True,True,False,0],#19.54,17.191
#                [2,'6378LF,10m,60kRad_set1+3+4+5_offset',19.68,17.332,0,'60 kRad',True,True,False,0],#19.54,17.191
#                [2,'6378LF,10m,60kRad_set1+3+4_offset',19.68,17.332,0,'60 kRad',True,True,False,0],#19.54,17.191
                #[2,'6378LF,10m,60kRad_set1+3+5_offset',19.68,17.332,0,'60 kRad',True,True,False,0],#19.54,17.191
#                [2,'6378LF,10m,60kRad_set-1+5_offset',19.68,17.332,0,'60 kRad',True,True,False,0],#19.54,17.191
                
                #[3,'6378Lf,10m,100kRad_set-1',19.68,16.039,0,'100 kRad',True,True,True,0],
#                [3,'6378Lf,10m,100kRad_set-2',19.68,16.179,0,'100 kRad',False,True,False,0],              
                #[3,'6378Lf,10m,100kRad_set-3',19.68,16.039,0,'100 kRad',False,True,False,0],
#                [3,'6378Lf,10m,100kRad_set-4',19.68,16.039,0,'100 kRad',False,True,False,0],
                #[3,'6378Lf,10m,100kRad_set-5',19.68,16.039,0,'100 kRad',False,True,False,0],
                #[3,'6378Lf,10m,100kRad_set-1+3+5',19.68,16.039,0,'100 kRad set 1+2+3',True,True,False,0],
                #[3,'6378Lf,10m,100kRad_set-1+2+3+5',19.68,16.039,0,'100 kRad set 1+2+3',True,True,False,0], 
                #[3,'6378Lf,10m,100kRad_set-5',19.68,16.039,0,'100 kRad',False,True,False,0],
#                ['CombinedData',19.54,16.039,0,'100 kRad',True,True,False,0],


#                [4,'6378LF,10m,100kRad_set-1',19.69,17.25,0,'6.5m',True,True,True,0],
#                [4,'6378LF,10m,100kRad_set-2',19.69,17.25,0,'6.5m',False,True,False,0],
#                [4,'6378LF,10m,100kRad_set-7',19.69,17.25,0,'6.5m',False,True,False,0],
#                [4,'6378LF,10m,100kRad_set-8',19.69,17.25,0,'6.5m',False,True,False,0],
                [4,'6378LF,10m,100kRad_set-1+2+7+8',19.69,17.25,0,'6.5m',True,True,False,0],
                ]

# Parameters in fileNameData:               
#   folderPath = path[fileNameData[0]]   (Index of folder path)
#   fileName = fileNameData[1]           (Name of .txt file with data)
#   maxPower = fileNameData[2]           (Gain value of fier before irradiation)
#   minPower = fileNameData[3]           (Gain value of fiber immediately after irradiation)
#   t0 = fileNameData[4]                 (Value of initial time, always zero, but could not be)
#   radiation[j] = fileNameData[5]       (Name of data set that will be featured in the final figure)
#   getFit[j] = fileNameData[6]          (Boolean to calculate fit, or simply show data as in in figure)
#   PremadeDf = fileNameData[7]          (Boolean to determine way of reading data, False = raw data from power mete, True = Data from Data_merging_v2.py)
#   showFutur = fileNameData[8]          (Boolean to allow futur prediction to be made and shown)
#   offset = fileNameData[9]             (Offsets data in power axis by this amount (corrects for offset between Jigs))




save = True            # Saves plots as .png and r2 in .xls
showMaxPower = True    # Show the maximum power value in graph
futurMuliplier = 9     # Factor to multuply orginal dataset length
setMaxValue = True     # Fit curve with max value parameter already set (This should always be true), has no use anymore, but part of the code that defines hte math functions requires it
getTimeData = True     # Define time in dataframe by difference between points, instead of fixed interval
multiPlot = True       # True = puts all sets in same figure, False = prints seperated figures for each set
multiPlotName = 'Name' # Old parameter, not supposed to do anything, but I'm afraid to delete it
combinePlots = False   # Old parameter, not supposed to do anything, but I'm afraid to delete it
unit = 'dBm'           # Old parameter, not supposed to do anything, but I'm afraid to delete it
#unit = 'mW'           # ^^ (The infrastructure to show graphs in mW instead od dBm is still there, but probably isn't functional anymore)
yLim = (16,19.8)       # Set y limit of graphs, if len(yLim) =, this will deterimne only the bottom limit, and the top limit will be the maxValue (must be a tuple with len = 1 or 2)
setXLim = False        # Determine wether to set limits of x axis or not
xLim = (80,120)        # if ^^ is True, the this will be the limits of x axis
GridOn = False         # Determine wether to put a grid on graphs, or not.

fitMultipleDataWithSameParameter = False #fits multiple data sets with the same parameter

#*****************************************************************************#
#* This script is split in two functionalities, depending on the value of    *#
#* fitMultipleDataWithSameParameter
#*
#*


###=========================================================================================== 

#define function for fitting

if setMaxValue:
    ArcTanFunc = lambda time,a,b : (maxPower*2/np.pi)*np.arctan(a*(time - b))
    InvFunc = lambda time,a,b: a/(time-b) + maxPower
    Logfunc = lambda time,a,b: a*np.log(time) +b
    ExpsumFunc = lambda time,a,b,c,d: maxPower + np.exp(a*(time-b))-np.exp(c*(time-d))
    WeibullFunc = lambda time,a,b: minPower + (maxPower-minPower)*(1- np.exp(-1*np.power((a*(time-t0)),b)))
    AccLossFunc_lambda = lambda time,b: minPower + (maxPower-minPower)*np.exp(-1*np.power((Lambda*(time-t0)),b))#0.67-0.89
    AccLossFunc_alpha = lambda time,a: minPower + (maxPower-minPower)*np.exp(-1*np.power((a*(time-t0)),Alpha))#-0.215 - -0.244
    
    Function = [#['Log function', Logfunc,2,[1,1],'dBm'],
                #['ArcTan function', ArcTanFunc,3,[2,3],'dBm'],
                #['Inverse function', InvFunc,3,[-2,-5],[-300,-16],['a','b']],
                
                ['Weibull function',WeibullFunc,2,[0.09,0.23],[-300,-16],['Lambda','Alpha']],  #(0.09,-0.23)   6.5 = 0.166,1
                #['WeibullFunc_Lambda= {}'.format(Lambda),AccLossFunc_lambda,1,[0.137],[-300,-16]], 
                #['WeibullFunc_Alpha = {}'.format(Alpha),AccLossFunc_alpha,1,[0.07],[-300,-16]],
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
 
# Parameters in Function:
# Function[0] = name of the function as it will appear in figure
# Function[1] = variable containing the actual mathematical function
# Function[2] = number of unkown variables (not used anymore, but I'm afraid to delete it)
# Function[3] = initial guesses of varibales
# Function[4] = initial guesses f varibles if using mW data (not used anymore, but I'm afraid to delete it)
# Function[0] = names of variable as they will appear in the figure
    
###===========================================================================================   
    
# Function used in plots to generate parameter boxes    
def parBox(ax,p,parName,x,y):
    #*************************************************************************#
    #* This function generates the boxes with information about the optimal  *#
    #* optimal parameter, for the figures                                    *#
    #*************************************************************************#
    
#    parName = ['lambda','alpha','c','d','e'] (not used anymore but i'm afraid to delete)
    string = ''
    for i in range(0,len(p)):
        string = string + parName[i] + '={'+'{}:.4f'.format(i)+'} '
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)

    ax.text(x,y, 'Parameters : \n '+string.format(*p), fontsize=12,
            verticalalignment='top', bbox=props)       
    
###===========================================================================================


    
def getDataFrame(plotData):
    #*************************************************************************#
    #* This function is not used anymore, but I'm afraid to delete it,       *#
    #* It create a .txt file and writes the content of ploData               *#
    #*************************************************************************#
    for i in range(0,len(plotData)):
        for j in range(0,len(plotData[i])):
            for k in range(0,len(plotData[i][j])):
                file = open(folderPath + "data"+str(i)+str(j)+str(k)+".txt","w")
                dataset = np.transpose(plotData[i][j][k])
                for l  in range(0,len(dataset)):
                    file.write("{} ".format(str(dataset[l][0])) + "{} ".format(str(dataset[l][1])) + "\n") 
                file.close() 
    
###===========================================================================================




#  WTF does this do !?!??  
#* I believe this is old code that is now obsolete, but I'm afraid to delete it                
titleVec = np.empty(len(Function), dtype = 'U50')
r2Vec = np.empty(len(Function))

for i in range(0,len(Function)):
    
    titleVec[i]  = Function[i][0]
    #print(Function[i][0])
r2data = {'Functions' : titleVec}            
#*****************************************************************************





# The main script starts here:
if fitMultipleDataWithSameParameter == False:
    
    # main loop start here 
     # j loops in the different files
     # i loops in the different functions  
    plotData = []
    pMat = []
    #radiation = []
    radiation = [['            ']]*len(fileNameData)
    getFit = np.zeros(len(fileNameData), dtype=bool)
    r2Mat = np.ones([len(fileNameData),len(Function)])
    #r2Mat = []       
    for j in range(0, len(fileNameData)):
        
        print(fileNameData[j][0])
        folderPath = path[fileNameData[j][0]]
        fileName = fileNameData[j][1]
        maxPower = fileNameData[j][2]
        minPower = fileNameData[j][3]
        t0 = fileNameData[j][4]
        radiation[j] = fileNameData[j][5][0:8]
    #    radiation = fileNameData[j][4]
        getFit[j] = fileNameData[j][6]
        PremadeDf = fileNameData[j][7]
        showFutur = fileNameData[j][8]
        offset = fileNameData[j][9]
           
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
            
        # Find and save best fit
        pVec = []
        plotDataTemp = []  
        if getFit[j]: 
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
                p,cov = curve_fit(func, time, power, p0, maxfev=20000)
                
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
                    fitTime = np.linspace(time[0],time[-1],10000)
                    fit = func(fitTime, *p)
                if j == 0 :
                    maxPowerTime = fitTime
                    
                # Calls the plot function or prepares the multiplot data  
        #        if multiPlot:  
                tempList = [np.array([time,power]),np.array([fitTime,fit])]
                plotDataTemp.append(tempList)
        #            print(r2Vec)
                    
        #        else:
        
        #            plt.close(ax)
                        
                pVec.append(p.tolist()[:])  
                    
        #    r2Mat.append(r2Vec[:])
            pMat.append(pVec[:])      
            r2Mat[j] = r2Vec
            plotData.append(plotDataTemp)    
        
        else:
            pMat.append([0])      
            r2Mat[j] = [0]
            plotData.append([[np.array([time,power])]])    
    
    
    
    
    
    
    
    
    ###############################################################################
    
    if multiPlot:
        # i is function index
        # j is data set index
        
        for i  in range(0,len(plotData[0])):
            #title  = Function[i][0]
            title  = '6378LF,10m,60kRad  - Weibull function'
            parName = Function[i][5]
            ax = plt.figure()
    
            legendData = []
            legendFit = []
        
            for j in range(0,len(plotData)):
                
                time    = plotData[j][i][0][0]
                power   = plotData[j][i][0][1]
                if getFit[j]:
                    fitTime = plotData[j][i][1][0]
                    fit     = plotData[j][i][1][1]
                r2 = r2Mat[i][0]
                folderPath = path[fileNameData[j][0]]
                fileName = fileNameData[j][1]
                
                p = pMat[j][i]
                
                plt.scatter(plotData[j][i][0][0],plotData[j][i][0][1],s=1) # Data
                if getFit[j]:
                    plt.plot(plotData[j][i][1][0],plotData[j][i][1][1]) # Fit  
                    if radiation == []:
                        legendFit.append('{}-power'.format(j+1))
                        legendData.append('{0}-fit : r2 = {1:.4f}'.format(j+1,r2Mat[j][i]))
                    else:
                        legendFit.append('{}-power'.format(radiation[j]))
                        legendData.append('{0}-fit : r2 = {1:.4f}'.format(radiation[j],r2Mat[j][i]))
                    
                    parBox(ax,np.array(pMat[j][i]),parName,.92,0.5-(0.15*j))
            
    
            legendVec = []
            legendVec.extend(legendData)
            legendVec.append('max power')
            legendVec.extend(legendFit)
            
            
            if showMaxPower:
              plt.plot(maxPowerTime, maxPower*np.ones(len(maxPowerTime)), label = 'maxPower')
                    
            bottom, top = plt.ylim()  # return the current ylim
     
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
            
            if GridOn:
                plt.grid(linestyle='--', linewidth=0.5)        
               
            
    
        if save:
            if os.path.exists(folderPath + 'multiPlot') == False:
    #                os.mkdir(title )
                os.mkdir(folderPath + 'multiPlot')
            plt.savefig(folderPath + 'multiPlot' +'\\'+ unit+title + '.png', bbox_inches="tight")
    #            plt.savefig(folderPath + 'multiPlot' +'\\'+'Parameter\\'+ unit+title+'_'+str(Alpha) + '.png', bbox_inches="tight")
    
    
    else:
        for i  in range(0,len(Function)):
            for j in range(0,len(plotData)):
                
                print('alli')
                time    = plotData[j][i][0][0]
                power   = plotData[j][i][0][1]
                fitTime = plotData[j][i][1][0]
                fit     = plotData[j][i][1][1]
                r2 = r2Mat[i][0]
                folderPath = path[fileNameData[j][0]]
                fileName = fileNameData[j][1]
                
                p = pMat[j][i]
                funcName = Function[i][0]
                parName = Function[i][5]
                
                ax = plt.figure()
    #            if unit == 'mW':
    #                plotFunc(ax,title,time,powerMW,maxPowerMW,fitTime,fit,r2Mat[i][0],
    #                         folderPath,fileName,showMaxPower,showFutur,unit)       
    #            else:
    #                plotFunc(ax,title,time,power,maxPower,fitTime,fit,r2Mat[i][0],
    #                         folderPath,fileName,showMaxPower,showFutur,unit)
    
                title = fileName + ' - ' + funcName   
                plt.title(title)
                plt.scatter(time, power, label = 'power',s=0.1,c='red')
                if showMaxPower:
                    plt.plot(fitTime, maxPower*np.ones(len(fitTime)), label = 'maxPower')
                if getFit[j]:
                    plt.plot(fitTime, fit, label  = 'fit')
                
                plt.legend(('max power','fit-{0} : r2 = {1:.4f}'.format(radiation[j],r2),'power-{0}'.format(radiation[j])),
                           loc = 'lower right',bbox_to_anchor=(1.4, -0.02))
                
                bottom, top = plt.ylim()  # return the current ylim
                plt.ylim((yLim, top)) 
            
                plt.xlabel('Time [Hours]')
                plt.ylabel('Power [{}]'.format(unit))
                
                parBox(ax,p,parName,0.925,0.5)         
                if GridOn:
                    plt.grid(linestyle='--', linewidth=0.5)                   
                    
                    
                if save:
                    if os.path.exists(folderPath + fileName) == False:
                        os.mkdir(folderPath + fileName )
                    plt.savefig(folderPath + fileName +'\\'+ title + '.png', bbox_inches="tight")
    
    #            plt.close(ax)




else:
    #*************************************************************************#
    #* This part is the average fitting of multipule datasets with the same  *#
    #* parameter. Mathematicla function and the intitial guesses are defined *#
    #* differently
    #*************************************************************************#

    # Initil guess, the script will loop thourgh them to unsure no local maxima, but the True maximum
    LambdaVec = (0.001 , 0.0015, 0.002 , 0.0025, 0.003 , 0.0035, 0.004)#np.arange(0.060,0.091,0.005) 
    AlphaVec = (0.18,0.19, 0.2 , 0.21, 0.22, 0.23, 0.24, 0.25, 0.26,) #np.arange(-0.200,-0.280,-0.01)

    readData = True     # Reads data from .txt files (should be True, but not neceseary if data is already in console)
    saveGraphPNG = True # Saves plots as .png 
    
    maxP = 0
    minP = 0
#    Weibull  = lambda time,Lambda,Alpha: minP + (maxP-minP)*(1 - np.exp(-1*np.power((Lambda*(time-t0)),Alpha)))
#    Function = ['AccLoss function',Weibull ,2,[0.09,-0.23],[-300,-16]]
    
    # This function is defined with MaxP and MinP as proper inputs, this is not the case of the earlier definition
    Weibull  = lambda time,Lambda,Alpha,maxP,minP,: minP + (maxP-minP)*(1 - np.exp(-1*np.power((Lambda*(time-t0)),Alpha)))


    ###########################################################################
    def r2AverageFunction(par,time,data,fit,findPar):
        #**********************************************************************#
        #* This funciton is used for the second functionality of the script,  *#
        #* to fit the multiple data sets with the same parameters. This is    *#
        #* the function that will be minimize if findPar is True, the value   *#
        #* to be minimized is 1- average r2, which will maximize r2. If       *#
        #* findPar is False, it will return a vector containing all individual*#
        #* r2 values for use in the figures                                   *#
        #**********************************************************************#
        

        Lambda= par[0]
        Alpha= par[1]
        
#        FVU = np.empty(len(data)) I believe this can be delete without problem
        r2vec = np.empty(len(data))
        
    #    print('momo')
        for i in range(0,len(data)):
    #        print(i)
            fit[i] = Weibull(time[i],Lambda,Alpha,maxPower[i],minPower[i])
               
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
    
    

        
    
        
    
        
    
    
    
    
    
    # loop to fit with different initial guesses
    for i in LambdaVec:
        for k in AlphaVec:
            
            print('Alpha = {0} - Lambda = {1}'.format(i,k))
            initialGuess = (i,k)
            
            res = minimize(r2AverageFunction,initialGuess,args=(time,data,fit,True),bounds=[(-2,10),(-2,10)])
            
            par = res['x']
            avgR2 = 1- res['fun']
            
            fit     = [None]*len(fileNameData)
            fitTime = [None]*len(fileNameData)
            
            R2Vec = r2AverageFunction(par,time,data,fit,False)
            
            
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
            plt.legend(['0 kRad','30 kRad - r2 = {:.5f}'.format((R2Vec)[0]),'60 kRad - r2 = {:.3f}'.format((R2Vec)[1]),'100 kRad - r2 = {:.3f}'.format((R2Vec)[2])])
            plt.xlabel('Time [Hours]')
            plt.ylabel('Power [dBm]')
            
            title = '6378LF,10m - Weibull function - Initial Guess = {0} - Average R2 = {1:.4f}'.format(initialGuess,avgR2)
            
            
            # Save
            if saveGraphPNG:
                if os.path.exists(folderPath + 'AverageR2') == False:
                    os.mkdir(folderPath + 'AverageR2')
                plt.savefig(folderPath + 'AverageR2' +'\\'+title +'lololol'+ '.png', bbox_inches="tight")
        
    
    
