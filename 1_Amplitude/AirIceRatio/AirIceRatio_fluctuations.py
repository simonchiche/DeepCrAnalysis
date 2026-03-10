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
    #if(theta==10.0): continue
    Traces_C, Traces_G, Pos = Shower.traces_c, Shower.traces_g, Shower.pos
    Nlay, Nplane, Depths = Shower.GetDepths()
    #Traces_tot = Shower.CombineTraces()

    # =============================================================================
    #                                Filter
    # =============================================================================

    Filter = False
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


SelE = 0.316
SelZen = 0.0
SelDepth = 3116

# Air/Ice radiation energy ratio per channel vs theta and depth
PlotAirIceEradRatiovsTheta(Eradair_allsims, Eradice_allsims, Depths, SelE, SelZen, OutputPath)


def PlotAirIceEradRatiovsThetavsE(Eradair_allsims, Eradice_allsims, SelDepth, OutputPath):

    EnergyAll = np.unique(Eradair_allsims[:,5])    
    
    for i in range(len(EnergyAll)):
        sel = (Eradair_allsims[:,4] == SelDepth) & (Eradair_allsims[:,5] == EnergyAll[i])
        
        #EradAirIceRatio_x = Eradair_allsims[sel][:,0]/ Eradice_allsims[sel][:,0]
        #EradAirIceRatio_y = Eradair_allsims[sel][:,1]/ Eradice_allsims[sel][:,1]
        #EradAirIceRatio_z = Eradair_allsims[sel][:,2]/ Eradice_allsims[sel][:,2]
        EradAirIceRatio_tot = Eradair_allsims[sel][:,3]/ Eradice_allsims[sel][:,3]

        arg = np.argsort(Eradair_allsims[sel][:,6])
        plt.errorbar(Eradair_allsims[sel][:,6][arg], EradAirIceRatio_tot[arg], yerr=0.35*EradAirIceRatio_tot[arg]) #, label ="E= $%.2f$ EeV" %EnergyAll[i])
        #plt.plot(Eradair_allsims[sel][:,6][arg], EradAirIceRatio_x[arg], label ="$E^{rad, air}_x/E^{rad, ice}_x$")
        #plt.plot(Eradair_allsims[sel][:,6][arg], EradAirIceRatio_y[arg], label ="$E^{rad, air}_y/E^{rad, ice}_y$")
        #plt.plot(Eradair_allsims[sel][:,6][arg], EradAirIceRatio_z[arg], label ="$E^{rad, air}_z/E^{rad, ice}_z$")
    #plt.scatter(Erad_allsims[sel][:,6], Erad_allsims[sel][:,3], label ="$E_{rad}-tot$")
    plt.legend(["$E=10^{16.5}\,$ eV", "$E=10^{17.0}\,$ eV", "$E=10^{17.5}\,$ eV"])

    plt.axhline(y=1.0, color='k', linestyle='--', linewidth=2.0)
    plt.yscale("log")
    #plt.ylim(0.01, 1e4)
    plt.ylabel("$E_{rad}^{air}/E_{rad}^{ice}\,$[50-1000 MHz]")
    plt.xlabel("Zenith [Deg.]")
    #plt.title("$|z| =%d$ m" %(SelDepth))
    plt.title("Depth =100 m", fontsize=14)
    plt.grid()
    plt.savefig(OutputPath + "air_ice_ratio_vs_theta_vsE_z%d.pdf" %SelDepth, bbox_inches = "tight")
    plt.show()

    return

SelDepth = 3156
PlotAirIceEradRatiovsThetavsE(Eradair_allsims, Eradice_allsims, SelDepth, OutputPath)

