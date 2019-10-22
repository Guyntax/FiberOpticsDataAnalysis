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

The main script containts two main functionalities:
	-Fit a model to existing data, one by one
	-Fit a model multiple data sets with the same parameter.


The first one loops through though all models and all data and produces a graph for each.
Multipule models can therefore be tried with mmany different dataset.
The Muliplot boolean allows all data and their fits too appear in the same figure.
If this is done, it will produce exactly one figure per model being tried.

The second optimizes the average value of all R2 score (coefficient of determination), 
instead of optimiszing each one idividually, therefore it returns the same parameters
for all datasets. This is usefull for the Weibull function, which appear to be the best model.

Both of these functionalities read the .txt files in the same way with two option.
	-it may read raw data from the power meter étxt files, in which the time is a timestamp
	-it may read processed data where the time is a value in hours that start at t_0, whose value is ussally 0