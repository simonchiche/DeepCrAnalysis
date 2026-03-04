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


def PlotMeanHpolVpolAirEradRatiovsThetavsE(Shower, Erad_allsims, SelDepth, title, OutputPath, Save, ylowlim=None, yhighlim=None):

    EnergyAll = np.unique(Erad_allsims[:,5])  
    Gdeep = Shower.glevel - SelDepth
    HVratioAll = dict()
    for i in range(len(EnergyAll)):
        sel = (Erad_allsims[:,4] == SelDepth) & (Erad_allsims[:,5] == EnergyAll[i])
        
        EradHpol_tot = (Erad_allsims[sel][:,0] + Erad_allsims[sel][:,1])
        EradVpol_tot = Erad_allsims[sel][:,2]
        EradHpoleVpoleRatio_tot = EradHpol_tot/EradVpol_tot

        arg = np.argsort(Erad_allsims[sel][:,6])
        HVratioAll[i] = EradHpoleVpoleRatio_tot[arg]
        #plt.plot(Erad_allsims[sel][:,6][arg], EradHpoleVpoleRatio_tot[arg], label ="$E= %.2f$ EeV" %EnergyAll[i])
    HVratioAllArr = np.vstack([HVratioAll[0], HVratioAll[1]]) #np.vstack([HVratioAll[0], HVratioAll[1], HVratioAll[2]])
    HVratiomean, HVratiostd =  np.mean(HVratioAllArr, axis=0), np.std(HVratioAllArr, axis=0)
    #plt.yscale("log")
    print(HVratioAllArr)
    #plt.ylim(min(data)/5, max(data)*5)
    
    print(len(Erad_allsims[sel][:,6][arg]), len(HVratioAllArr), len(HVratiostd))
    if(title == "In-air"):
        plt.errorbar(Erad_allsims[sel][:,6][arg], HVratiomean, yerr = HVratiostd, label ="in-air", marker ="o", color = "#D62728")
    if(title == "In-ice"):
        plt.errorbar(Erad_allsims[sel][:,6][arg], HVratiomean, yerr = HVratiostd, label ="in-ice", marker ="o", color = "#4F81BD")
    plt.ylabel("$E_{\mathrm{rad}}^{\mathrm{Hpol}}/E_{\mathrm{rad}}^{\mathrm{Vpol}}\,$[50-1000 MHz]")
    plt.xlabel("Zenith [Deg.]")
    plt.grid()
    plt.legend()
    plt.axvspan(0, 20, color='grey', alpha=0.4, zorder=0)
    if(ylowlim and yhighlim):
        plt.ylim(ylowlim, yhighlim)
    #plt.ylim(1, 4)
    plt.title(title + ", Depth =%d m" %(Gdeep), fontsize =14)
    plt.savefig(OutputPath + "_" + title + "mean_Hpol_over_Vpol_vs_E_vs_zenith_z%d.pdf"\
                 %SelDepth, bbox_inches = "tight") if Save else None
    plt.show()

    return


SelDepth = 3116
title = "In-air"
PlotMeanHpolVpolAirEradRatiovsThetavsE(Shower, Eradair_allsims, SelDepth, title, OutputPath, Save=True)

title = "In-ice"
PlotMeanHpolVpolEradRatiovsThetavsE(Shower, Eradice_allsims, SelDepth, title, OutputPath, Save= True,  ylowlim=1.0, yhighlim=4.0)
