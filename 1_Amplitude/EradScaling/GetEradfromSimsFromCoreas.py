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
from Modules.PlotErad import PlotEradThetaScaling, PlotEradDepthScaling, PlotEradEnergyScaling, PlotEradEScalingvsDepth,PlotAirIceEradRatiovsTheta, PlotAirIceEradRatiovsThetavsE, PlotHpoleVpoleEradRatiovsThetavsE
from Modules.ModuleGetCoreasMaps import GetCoreasTracesfromHDF5, PlotCoreasMaps
from MainModules.FormatFaerieOutput import Traces_cgs_to_si
#endregion

#region Path definition
SimDir = "PolarDenseDeepCr"  #"InterpSim"
WorkPath = os.getcwd()
BatchID = "PolarDense_Erad_filtered"
OutputPath = MatplotlibConfig(WorkPath, SimDir, BatchID)
#endregion
Save = False
SimpathAll = glob.glob("/Users/chiche/Desktop/DeepCrAnalysis/Simulations/DeepCrLibV1/*")

Eradair_allsims = []
Eradice_allsims = []

for simpath in SimpathAll:
    print(simpath.split("/")[-1])
    Shower = CreateShowerfromHDF5(simpath)
    Shower.traces_c = GetCoreasTracesfromHDF5(simpath)
    Shower.traces_c = Traces_cgs_to_si(Shower.traces_c)
    # =============================================================================
    #                              Load Traces
    # =============================================================================

    energy, theta, Nant = Shower.energy, Shower.zenith, Shower.nant
    if(theta==10.0): continue
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
        #Traces_G =Shower.filter_all_traces(Traces_G, fs, lowcut, highcut)

    # =============================================================================
    #                         Radiation energy
    # =============================================================================

    Eradair_allsims.append(Shower.GetEradFromSim(Traces_C))
    #Eradice_allsims.append(Shower.GetEradFromSim(Traces_G))
 
Eradair_allsims = np.concatenate(Eradair_allsims, axis =0)
#Eradice_allsims = np.concatenate(Eradice_allsims, axis =0)

# =============================================================================
#                             Plots
# =============================================================================

# In-air radiation energy vs depth
SelZen = 0
title = "In-air"
PlotEradDepthScaling(Eradair_allsims, SelZen, title, OutputPath)

# In-ice radiation energy vs depth
#SelZen = 0
#title = "In-ice"
#PlotEradDepthScaling(Eradice_allsims, SelZen, title, OutputPath)

# In-air radiation energy vs zenith angle
SelE = 0.316
title = "In-air"
PlotEradThetaScaling(Eradair_allsims, Depths, SelE, SelZen, title, OutputPath)

# In-ice radiation energy vs zenith angle
#SelE = 0.316
#title = "In-ice"
#PlotEradThetaScaling(Eradice_allsims, Depths, SelE, SelZen, title, OutputPath)

# In-air radiation energy vs primary energy
SelDepth = 3116
title = "In-air"
PlotEradEnergyScaling(Eradair_allsims, SelDepth, title, OutputPath)

# In-ice radiation energy vs primary energy
#SelDepth = 3116
#title = "In-ice"
#PlotEradEnergyScaling(Eradice_allsims, SelDepth, title, OutputPath)

# In-air Energy scaling as a function of depth
title = "In-air"
Eindex = 2 #E17.5/E17
PlotEradEScalingvsDepth(Eradair_allsims, Eindex, title, OutputPath)
Eindex = 1 #E17/E16.5
PlotEradEScalingvsDepth(Eradair_allsims, Eindex, title, OutputPath)

# In-ice Energy scaling as a function of depth
#title = "In-ice"
#Eindex = 2 #E17.5/E17
#PlotEradEScalingvsDepth(Eradice_allsims, Eindex, title, OutputPath)
#Eindex = 1 #E17/E16.5
#PlotEradEScalingvsDepth(Eradice_allsims, Eindex, title, OutputPath)

# Air/Ice radiation energy ratio per channel vs theta and depth
#PlotAirIceEradRatiovsTheta(Eradair_allsims, Eradice_allsims, Depths, SelE, SelZen, OutputPath)

# Total Air/Ice radiation energy ratio vs theta for different energy bins at fixed depth
#SelDepth = Depths[-2]
#PlotAirIceEradRatiovsThetavsE(Eradair_allsims, Eradice_allsims, SelDepth, OutputPath)

# Hpole/Vpole Erad air ratio vs theta for different energy bins
title = "In-air"
PlotHpoleVpoleEradRatiovsThetavsE(Eradair_allsims, SelDepth, title, OutputPath)

# Hpole/Vpole Erad ice ratio vs theta for different energy bins
#title = "In-ice"
#PlotHpoleVpoleEradRatiovsThetavsE(Eradice_allsims, SelDepth, title, OutputPath)

# (1) Do all the plots
# (2) Add right label, title and save
# (3) Adapt for new sims



# (1) Get the energy scaling for all depths Loop over theta: Erad2/Erad1 vs depth for air and ice emission
# (2) Get the air to ice emission ratio vs theta for the three energies
# (3) Get the air to ice emission ratio vs theta at 60 and 100 meters deep
# (4) Get the Hpol (mean of Eradx and Erady) to Vpol (Eradz) ratio vs theta for the different energy bins

### Beyond ###
## Maps
# Plot the Efield maps for the new simulations
## Erad
# Get the radiation energy results for the new simulations


## Polarization
# Write a script with histograms of the Hpole/Vpole ratio at the antenna level (1) for different simulations (zenith angles) at 10^17.5 eV 
# and (2) for all zenith angles but different energy bins
## Frequency spectra