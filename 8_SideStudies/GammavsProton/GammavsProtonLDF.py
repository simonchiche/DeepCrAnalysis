#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 02:21:35 2024

@author: chiche
"""
#SCRIPT NEEDS UPDATES!

# Modules import
#region Modules 
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
import glob
import sys
import pickle
from MainModules.ShowerClass import CreateShowerfromHDF5
from MainModules.PlotConfig import MatplotlibConfig
sys.path.append("/Users/chiche/Desktop/DeepCrSearch/Analysis/")
##from Modules.SimParam.GetLayoutScaling import GetDepthScaling, GetEnergyScaling
from scipy.interpolate import interp1d
import scipy
#from FunctionsPlotFluence import EfieldMap, PlotLDF, PlotTraces, plot_polarisation, PlotMaxTraces, PlotAllTraces, PlotLayer, PlotGivenTrace, PlotAllChannels, PlotSurfaceEz
from Modules.ModulePlotRadioSimExtent import PlotFillingFactor, PlotRadioSimExtent, PlotAirIceExtent, PlotMaxLDF
from scipy.interpolate import griddata
from datetime import datetime
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt
from Modules.ModuleSimExtent import GetRadioExtent, GetMaxLDF, GetCaracExtent
##from Modules.Fluence.FunctionsRadiationEnergy import GetFluence, GetRadiationEnergy
#endregion

#region Path definition
SimDir = "DeepCrLib"  #"InterpSim"
SimName = "Rectangle_Proton_0.0316_0_0_1"
WorkPath = os.getcwd()
simpath = "/Users/chiche/Desktop/DeepCrAnalysis/Simulations/DeepCrLibV1/"\
      + SimName + "_0.hdf5"
BatchID = ""
OutputPath = MatplotlibConfig(WorkPath, SimDir, BatchID)
#endregion
Save = False

DataPath= "/Users/chiche/Desktop/"
posx_MaxAirLDF_Gamma = np.loadtxt(DataPath + "GammaLDFdataPosAir.txt")
posx_MaxIceLDF_Gamma  = np.loadtxt(DataPath + "GammaLDFdataPosIce.txt")
Ex_MaxAirLDF_Gamma = np.loadtxt(DataPath + "GammaLDFdataEIce.txt")
Ex_MaxIceLDF_Gamma =  np.loadtxt(DataPath + "GammaLDFdataEAir.txt")

DataPath= "/Users/chiche/Desktop/"
posx_MaxAirLDF_Proton = np.loadtxt(DataPath + "ProtonLDFdataPosAir.txt")
posx_MaxIceLDF_Proton  = np.loadtxt(DataPath + "ProtonLDFdataPosIce.txt")
Ex_MaxAirLDF_Proton = np.loadtxt(DataPath + "ProtonLDFdataEIce.txt")
Ex_MaxIceLDF_Proton =  np.loadtxt(DataPath + "ProtonLDFdataEAir.txt")

plt.scatter(posx_MaxAirLDF_Proton[:,0], Ex_MaxAirLDF_Proton,label = "in-air proton")
plt.scatter(posx_MaxAirLDF_Gamma[:,0], Ex_MaxAirLDF_Gamma,label = "in-air gamma")
plt.yscale("log")
plt.xlabel("x [m]")
plt.ylabel("$E_{int}$ [$\mu Vs$/m]")
plt.title("Depth = 0 m")
plt.legend()
plt.show()