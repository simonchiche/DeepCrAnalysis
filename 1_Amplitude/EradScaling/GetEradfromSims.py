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
#endregion

#region Path definition
SimDir = "DeepCrLibV1"  #"InterpSim"
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

# In-air radiation energy vs depth
SelZen = 0
title = "In-air"
PlotEradDepthScaling(Eradair_allsims, SelZen, title, OutputPath)

# In-ice radiation energy vs depth
SelZen = 0
title = "In-ice"
PlotEradDepthScaling(Eradice_allsims, SelZen, title, OutputPath)

# In-air radiation energy vs zenith angle
SelE = 0.316
title = "In-air"
PlotEradThetaScaling(Eradair_allsims, Depths, SelE, SelZen, title, OutputPath)

# In-ice radiation energy vs zenith angle
SelE = 0.316
title = "In-ice"
PlotEradThetaScaling(Eradice_allsims, Depths, SelE, SelZen, title, OutputPath)

# In-air radiation energy vs primary energy
SelDepth = 3116
title = "In-air"
PlotEradEnergyScaling(Eradair_allsims, SelDepth, title, OutputPath)

# In-ice radiation energy vs primary energy
SelDepth = 3116
title = "In-ice"
PlotEradEnergyScaling(Eradice_allsims, SelDepth, title, OutputPath)

# In-air Energy scaling as a function of depth
title = "In-air"
Eindex = 2 #E17.5/E17
PlotEradEScalingvsDepth(Eradair_allsims, Eindex, title, OutputPath)
Eindex = 1 #E17/E16.5
PlotEradEScalingvsDepth(Eradair_allsims, Eindex, title, OutputPath)

# In-ice Energy scaling as a function of depth
title = "In-ice"
Eindex = 2 #E17.5/E17
PlotEradEScalingvsDepth(Eradice_allsims, Eindex, title, OutputPath)
Eindex = 1 #E17/E16.5
PlotEradEScalingvsDepth(Eradice_allsims, Eindex, title, OutputPath)


def PlotEradtotThetaScaling(Eradair_allsims, Eradice_allsims,  Depths, SelE, SelZen, title, OutputPath):
    #sel = (Erad_allsims[:,6] == SelZen) & (Erad_allsims[:,5] == SelE)
    
    for i in range(len(Depths)):

        sel = (Eradair_allsims[:,4] == Depths[i]) & (Eradair_allsims[:,5] == SelE)
        
        arg = np.argsort(Eradair_allsims[sel][:,6])
        plt.plot(Eradair_allsims[sel][:,6][arg], Eradair_allsims[sel][:,3][arg], label ="In-air")
        plt.plot(Eradice_allsims[sel][:,6][arg], Eradice_allsims[sel][:,3][arg], label ="In-ice")
        plt.yscale("log")
        #plt.ylim(min(data)/5, max(data)*5)
        plt.ylabel("$E_{rad} \, $[MeV]")
        plt.xlabel("Zenith [Deg.]")
        plt.legend()
        selDepth = 3216- Depths[i]
        plt.title( " $E=%.2f\,$ EeV Depth =%d m" %(SelE,selDepth), fontsize =12)
        plt.grid()
        plt.savefig(OutputPath + "_" + title + "_vs_zenith_E%.2f_z%d.pdf" %(SelE, Depths[i]), bbox_inches = "tight")
        plt.show()

    return

title ="Air_vs_Ice_vs_Theta"
PlotEradtotThetaScaling(Eradair_allsims, Eradice_allsims, Depths, SelE, SelZen, title, OutputPath)