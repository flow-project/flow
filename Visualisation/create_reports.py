"""
author: Sadman Ahmed Shanto
purpose: read and analyze data and create plots
"""
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import pandas as pd
import seaborn as sns

#different visualations commands
#sns.countplot(x='Many_service_calls', hue='Churn', data=df);
#df[["a1", "a2"]].plot(bins=30, kind="hist")


#directory where data.csv is stored
dataLoc = "/Users/sshanto/Vanderbilt/flow-master/examples/data/"
dataFile = sys.argv[1]
uniqueName = dataFile.split(".")[0]

df = pd.read_csv(dataLoc+dataFile)

def printEverythingAgainstTime():
    #if number
    for i in range(len(df.columns)):
        val = df.columns[i]
        df.plot.scatter(x="time",y=val) 
        plt.title("Plot of " +str(val) + " against time")
        plt.show()

printEverythingAgainstTime()
# plot everything wrt to time
# plot everything wrt to cars
# lamda function plots 
