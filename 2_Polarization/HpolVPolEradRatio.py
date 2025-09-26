#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 02:21:35 2024

@author: chiche
"""

# Modules import
#region Modules 
import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
import glob
import sys
import pickle
from MainModules.ShowerClass import CreateShowerfromHDF5
from MainModules.PlotConfig import MatplotlibConfig
from scipy.interpolate import interp1d
import scipy
from scipy.interpolate import griddata
from datetime import datetime
from scipy.optimize import curve_fit
from Modules.PlotErad import PlotEradThetaScaling, PlotEradDepthScaling, PlotEradEnergyScaling, PlotEradEScalingvsDepth,PlotAirIceEradRatiovsTheta, PlotAirIceEradRatiovsThetavsE, PlotHpolVpolEradRatiovsThetavsE, PlotMeanHpolVpolEradRatiovsThetavsE
#endregion

#region Path definition
SimDir = "FullDenseDeepCr" #"DeepCrLibV1"  #"InterpSim"
WorkPath = os.getcwd()
BatchID = "Erad_filtered"
OutputPath = MatplotlibConfig(WorkPath, SimDir, BatchID)
#endregion
Save = False
simpath = "/Users/chiche/Desktop/DeepCrAnalysis/Simulations/" + SimDir
SimpathAll = glob.glob(simpath + "/*")

Eradair_allsims = []
Eradice_allsims = []
Eradtot = []

for simpath in SimpathAll:
    print(simpath.split("/")[-1])
    Shower = CreateShowerfromHDF5(simpath)
    # =============================================================================
    #                              Load Traces
    # =============================================================================

    energy, theta, Nant = Shower.energy, Shower.zenith, Shower.nant
    Traces_C, Traces_G, Pos = Shower.traces_c, Shower.traces_g, Shower.pos
    Nlay, Nplane, Depths = Shower.GetDepths()
    #Traces_tot = Shower.CombineTraces()

    # =============================================================================
    #                                Filter
    # =============================================================================

    Filter = True
    if(Filter):
        fs, lowcut, highcut = 5e9, 50e6, 1e9
        #Traces_C_filtered =Shower.filter_all_traces(Traces_C, fs, lowcut, highcut)
        #Traces_G_filered =Shower.filter_all_traces(Traces_G, fs, lowcut, highcut)

        Traces_C =Shower.filter_all_traces(Traces_C, fs, lowcut, highcut)
        Traces_G =Shower.filter_all_traces(Traces_G, fs, lowcut, highcut)

    # =============================================================================
    #                         Radiation energy
    # =============================================================================

    Eradair_allsims.append(Shower.GetEradFromSim(Traces_C))
    Eradice_allsims.append(Shower.GetEradFromSim(Traces_G))
    Eradtot.append(Shower.GetRadiationEnergyGeneric(Traces_C))
 
Eradair_allsims = np.concatenate(Eradair_allsims, axis =0)
Eradice_allsims = np.concatenate(Eradice_allsims, axis =0)

# =============================================================================
#                             Plots
# =============================================================================

# Hpole/Vpole Erad air ratio vs theta for different energy bins
Save = True
SelDepth = 3116  # Depth in meters
title = "In-air"
PlotHpolVpolEradRatiovsThetavsE(Shower, Eradair_allsims, SelDepth, title, OutputPath, Save)

# Hpole/Vpole Erad ice ratio vs theta for different energy bins
SelDepth = 3116
title = "In-ice"
PlotHpolVpolEradRatiovsThetavsE(Shower, Eradice_allsims, SelDepth, title, OutputPath, Save, 1.0, 4.0)


title = "In-air"
PlotMeanHpolVpolEradRatiovsThetavsE(Shower, Eradair_allsims, SelDepth, title, OutputPath, Save=True)

title = "In-ice"
PlotMeanHpolVpolEradRatiovsThetavsE(Shower, Eradice_allsims, SelDepth, title, OutputPath, Save= True,  ylowlim=1.0, yhighlim=4.0)
