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
from Modules.PlotErad import PlotEradThetaScaling, PlotEradDepthScaling, PlotEradEnergyScaling, PlotEradEScalingvsDepth,PlotAirIceEradRatiovsTheta, PlotAirIceEradRatiovsThetavsE, PlotHpoleVpoleEradRatiovsThetavsE, PlotEradtotThetaScaling, GetMeanEradScalingVsE, PlotMeanEradScalingVsE, PlotEradIceEScalingvsDepth, PlotGroundParticleEVsZenith, PlotEradIcevsZenE, PlotEradIcevsEgroundPart, EradicevsZenE, GetEradvsEgroundPart, GetGroundParticleEnergy, GetEgroundPart_E
#endregion

#region Path definition
SimDir = "FullDenseDeepCr" #"DeepCrLibV1"  #"InterpSim"
WorkPath = os.getcwd()
BatchID = "Erad_filtered"
OutputPath = MatplotlibConfig(WorkPath, SimDir, BatchID)
#endregion
Save = True
simpath = "/Users/chiche/Desktop/DeepCrAnalysis/Simulations/" + SimDir
SimpathAll = glob.glob(simpath + "/*")

Eradair_allsims = []
Eradice_allsims = []
Eradtot = []
counter = 0
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

    #sys.exit(Shower.GetRadiationEnergyGeneric(Traces_C))
    Eradair_allsims.append(Shower.GetRadiationEnergyGeneric(Traces_C))
    Eradice_allsims.append(Shower.GetRadiationEnergyGeneric(Traces_G))
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

# Air, Ice radiation energy vs zenith angle

def PlotEradAirIceThetaScaling(Eradair_allsims, Eradice_allsims, Depths, SelE, SelZen, title, OutputPath):
    #sel = (Erad_allsims[:,6] == SelZen) & (Erad_allsims[:,5] == SelE)
    
    for i in range(len(Depths)):

        sel = (Eradair_allsims[:,4] == Depths[i]) & (Eradair_allsims[:,5] == SelE)
        
        arg = np.argsort(Eradair_allsims[sel][:,6])
        plt.plot(Eradair_allsims[sel][:,6][arg], Eradair_allsims[sel][:,0][arg], label ="$E^{\mathrm{rad}}_{x,\mathrm{air}}$", color="#0072B2", linestyle='dashed')
        plt.plot(Eradair_allsims[sel][:,6][arg], Eradair_allsims[sel][:,1][arg], label ="$E^{\mathrm{rad}}_{y,\mathrm{air}}$", color="#E69F00", linestyle='dashed')
        plt.plot(Eradair_allsims[sel][:,6][arg], Eradair_allsims[sel][:,2][arg], label ="$E^{\mathrm{rad}}_{z,\mathrm{air}}$", color="#CC79A7", linestyle='dashed')
        plt.plot(Eradice_allsims[sel][:,6][arg], Eradice_allsims[sel][:,0][arg], label ="$E^{\mathrm{rad}}_{x,\mathrm{ice}}$", color="#0072B2")
        plt.plot(Eradice_allsims[sel][:,6][arg], Eradice_allsims[sel][:,1][arg], label ="$E^{\mathrm{rad}}_{y,\mathrm{ice}}$", color="#E69F00")
        plt.plot(Eradice_allsims[sel][:,6][arg], Eradice_allsims[sel][:,2][arg], label ="$E^{\mathrm{rad}}_{z,\mathrm{ice}}$", color="#CC79A7")
        #plt.scatter(Erad_allsims[sel][:,6], Erad_allsims[sel][:,3], label ="$E_{rad}-tot$")
        plt.yscale("log")
        #plt.ylim(min(data)/5, max(data)*5)
        plt.ylabel("$E_{\mathrm{rad}} \, $[MeV]")
        plt.xlabel("Zenith [Deg.]")
        plt.legend(ncol=2, framealpha=0.8)
        plt.grid(alpha=0.3)
        plt.title(" $E=10^{17.5}\,$eV, Depth =%d m" %(3216-Depths[i]), fontsize=14)
        plt.savefig(OutputPath + "_" + title + "_vs_zenith_E%.2f_z%d.pdf" %(SelE, Depths[i]), bbox_inches = "tight")
        plt.show()

    return
title ="AirIce"
PlotEradAirIceThetaScaling(Eradair_allsims, Eradice_allsims, Depths, SelE, SelZen, title, OutputPath)

# In-air radiation energy vs primary energy
SelDepth = 3116
title = "In-air"
PlotEradEnergyScaling(Eradair_allsims, SelDepth, title, Shower, OutputPath)


# Mean Erad scaling vs E
SelDepth = 3116
X, y_mean, y_std, y_mean_ice, y_std_ice = \
    GetMeanEradScalingVsE(Eradair_allsims, Eradice_allsims, SelDepth, title, OutputPath)

PlotMeanEradScalingVsE(X, y_mean, y_std, y_mean_ice, y_std_ice, SelDepth, Shower, OutputPath)

# In-ice radiation energy vs primary energy
SelDepth = 3116
title = "In-ice"
PlotEradEnergyScaling(Eradice_allsims, SelDepth, title, Shower, OutputPath)

# In-air Energy scaling as a function of depth
title = "In-air"
Eindex = 2 #E17.5/E17
PlotEradEScalingvsDepth(Eradair_allsims, Eindex, title, OutputPath)

title = "In-air"
Eindex = 1 #E17/E16.5
PlotEradEScalingvsDepth(Eradair_allsims, Eindex, title, OutputPath)

title = "In-ice"
Eindex = 1 #E17/E16.5
PlotEradEScalingvsDepth(Eradice_allsims, Eindex, title, OutputPath)


# In-ice Energy scaling as a function of depth
title = "In-ice"
Eindex = 2 #E17.5/E17
PlotEradIceEScalingvsDepth(Eradice_allsims, Eindex, title, OutputPath)
Eindex = 1 #E17/E16.5
PlotEradIceEScalingvsDepth(Eradice_allsims, Eindex, title, OutputPath)

SelE = 0.316
title ="Air_vs_Ice_vs_Theta"
PlotEradtotThetaScaling(Eradair_allsims, Eradice_allsims, Depths, SelE, SelZen, title, OutputPath)




################ Test Ground Particle Energy Scaling vs Depth #####################


# Ground Particle Energy vs Zenith Angle at fixed Primary Energy
DataAll = \
    glob.glob("/Users/chiche/Desktop/DeepCrAnalysis/8_SideStudies/ShowerPhysics/Data/GroundParticleFiles/*")
EGroundPart, EprimaryAllpart, ZenithAllpart = GetGroundParticleEnergy(DataAll)
SelE = max(EprimaryAllpart)


def GetEgroundPart_E(EGroundPart, EprimaryAllpart, ZenithAllpart, SelE):
    Ebins = np.unique(EprimaryAllpart)

    ZenE = ZenithAllpart[EprimaryAllpart==SelE]
    EGroundPartE = EGroundPart[EprimaryAllpart==SelE]

    argzensort = np.argsort(ZenE)
    ZenE = ZenE[argzensort]
    EGroundPartE = EGroundPartE[argzensort]

    return ZenE, EGroundPartE


ZenE, EGroundPartE = GetEgroundPart_E(EGroundPart, EprimaryAllpart, ZenithAllpart, SelE)
PlotGroundParticleEVsZenith(ZenE, EGroundPartE, SelE)

# In-ice radiation  Energy vs Zenith Angle at fixed Primary Energy
#SelE = max(EprimaryAllpart)

def EradicevsZenE(Eradice_allsims, Depths, SelE):
    
    Ebins = np.unique(Eradice_allsims[:,5])

    sel = (Eradice_allsims[:,4] == min(Depths))  & (Eradice_allsims[:,5] ==SelE)
    arg = np.argsort(Eradice_allsims[sel][:,6])

    ZenE_Erad = Eradice_allsims[sel][:,6][arg]
    EradiceE = Eradice_allsims[sel][:,3][arg]

    return ZenE_Erad, EradiceE


ZenE_Erad, EradiceE = EradicevsZenE(Eradice_allsims, Depths, SelE)
PlotEradIcevsZenE(ZenE_Erad, EradiceE)

# In-ice radiation Energy vs Ground Particle Energy at fixed Primary Energy
EGroundPartE, EfieldIceE, X, Ylinear = GetEradvsEgroundPart(EGroundPartE, EradiceE)

def PlotEradIcevsEgroundPart(EGroundPartE, EfieldIceE, OutputPath):
    from scipy.optimize import curve_fit
    def ModelFunc(x, a,b):
        return  a*x**b
    popt, pcov = curve_fit(ModelFunc, EGroundPartE, EfieldIceE)

    plt.scatter(EGroundPartE, EfieldIceE, marker='*', label="$E_{\mathrm{rad}}^{\mathrm{ice}}$", s=50)
    #plt.plot(X, Ylinear+0.1e-7, 'r--', label='Linear scaling')
    plt.plot(EGroundPartE, ModelFunc(EGroundPartE, *popt), label= r"$a (E_{\mathrm{rad}}^{\mathrm{part}})^{b}$", color="red")
    plt.xlabel('$E_{\mathrm{part}}^{\mathrm{ground}}\,[\mathrm{GeV}]$')
    plt.ylabel('$\sqrt{E_{\mathrm{rad}}^{\mathrm{ice}}\,[MeV]}$')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.savefig(OutputPath + "EfieldIce_vs_EgroundPart.pdf", bbox_inches = 'tight')
    plt.show()

    a_err, b_err = np.sqrt(np.diag(pcov))
    yfit = ModelFunc(EGroundPartE, *popt)
    ss_res = np.sum((EfieldIceE - yfit)**2)
    ss_tot = np.sum((EfieldIceE - np.mean(EfieldIceE))**2)
    r2 = 1 - ss_res/ss_tot


    print("Fit parameters: a = %.3e, b = %.3f" % (popt[0], popt[1]))
    print(a_err, b_err, r2)
    return  a_err, b_err, r2

a_err, b_err, r2 = PlotEradIcevsEgroundPart(EGroundPartE, EfieldIceE, OutputPath)





## Fixer les unités
# obtenir les plots sur le zenith
# Air/ice ratio