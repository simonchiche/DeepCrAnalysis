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
from  MainModules.PlotConfig import MatplotlibConfig
##from Modules.SimParam.GetLayoutScaling import GetDepthScaling, GetEnergyScaling
from scipy.interpolate import interp1d
import scipy
##from Modules.Fluence.FunctionsGetFluence import  Norm, LoadTraces, GetPeakTraces, Traces_cgs_to_si, GetDepths, CorrectScaling, CombineTraces, CorrectLength, GetIntTraces, GetIntTracesSum, GetRadioExtent
from Modules.FunctionsPlotFluence import EfieldMap, PlotLDF, PlotTraces, plot_polarisation, PlotMaxTraces, PlotAllTraces, PlotLayer, PlotGivenTrace, PlotAllChannels, PlotSurfaceEz, RemoveCoreAntennas
##from CleanCoreasTraces import CleanCoreasTraces
##from Modules.SimParam.PlotRadioSimExtent import PlotFillingFactor, PlotRadioSimExtent, PlotAirIceExtent
from scipy.interpolate import griddata
from datetime import datetime
from scipy.optimize import curve_fit
from Modules.ModulePlotAmplitudeDistrib import PlotAmplitudeDistribution
from Modules.ModuleGetAmplitudeDistrib import GetAmplitudeDistribution
##from Modules.Fluence.FunctionsRadiationEnergy import GetFluence, GetRadiationEnergy
#endregion

#region Path definition
SimDir = "DeepCrLib"  #"InterpSim"
WorkPath = os.getcwd()
BatchID = "Log10"
OutputPath = MatplotlibConfig(WorkPath, SimDir, BatchID)
#endregion
Save = False
SimpathAll = glob.glob("/Users/chiche/Desktop/DeepCrAnalysis/Simulations/DeepCrLibV1/*")

EnergyAll, ZenithAll, ExAirAll, EyAirAll, EzAirAll,  ExIceAll, EyIceAll, EzIceAll, HVratioAirAll, HVratioIceAll, PosAll = \
    ([] for _ in range(11))

for simpath in SimpathAll:
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
    #                           Get peak amplitude
    # =============================================================================

    ExC, EyC, EzC, EtotC = Shower.GetPeakTraces(Traces_C)
    ExG, EyG, EzG, EtotG = Shower.GetPeakTraces(Traces_G)
    #Extot, Eytot, Eztot, Etot_peak = Shower.GetPeakTraces(Traces_tot)

    # =============================================================================
    #                                 Get integral
    # =============================================================================

    ExC_int, EyC_int, EzC_int, EtotC_int, peakTime = Shower.GetIntTraces(Traces_C)
    ExG_int, EyG_int, EzG_int, EtotG_int, peakTime = Shower.GetIntTraces(Traces_G)
    #Ex_tot_int, Ey_tot_int, Ez_tot_int, Etot_int, peakTime = Shower.GetIntTraces(Traces_tot)

    # We restrict the analysis to deep antennas only
    #sel = Pos[:,2] == 3116
    HVratioAir = (np.array(ExC_int) + np.array(EyC_int))/(2*np.array(EzC_int))
    HVratioIce = (np.array(ExG_int) + np.array(EyG_int))/(2*np.array(EzG_int))

    HVratioAir=HVratioAir[~np.isnan]
    HVratioIce=HVratioIce[~np.isnan]
    for lst, val in zip([HVratioAirAll, HVratioIceAll, ExAirAll, EyAirAll, EzAirAll,  ExIceAll, EyIceAll, EzIceAll, EnergyAll, ZenithAll, PosAll],
                        [HVratioAir, HVratioIce, ExC_int, EyC_int, EzC_int, ExG_int, EyG_int, EzG_int, energy, theta, Pos]):
        lst.append(val)


HVratioAirAll, HVratioIceAll, ExAirAll, EyAirAll, EzAirAll, ExIceAll, EyIceAll, EzIceAll, EnergyAll, ZenithAll, PosAll =\
      map(np.array, [HVratioAirAll, HVratioIceAll, ExAirAll, EyAirAll, EzAirAll, ExIceAll, EyIceAll, EzIceAll, EnergyAll, ZenithAll, PosAll])



ZenithBins = np.unique(ZenithAll)

for i in range(len(ZenithBins)):

    sel = (ZenithAll == ZenithBins[i])
    HVratiozen_air = HVratioAirAll[sel].flatten()
    HVratiozen_ice = HVratioIceAll[sel].flatten()
    bin_edges = np.linspace(0, 10, 80) 

    plt.hist(HVratiozen_air, bin_edges, alpha=0.6, edgecolor='black')
    plt.xlabel('Hpole/Vpole')
    plt.ylabel('Nant')
    #plt.xlim(0,2000)
    plt.legend()
    #if(scale=="log"): plt.yscale("log")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.axvline(x=1, color='red', linestyle='--', linewidth=2)
    #plt.title("In-air emission")
    plt.title(r"In-air, $\theta =%.d^{\circ}$" %ZenithBins[i])
    plt.savefig("/Users/chiche/Desktop/HVratio/InAirFilteredHVratio_zen%.d.pdf" %ZenithBins[i], bbox_inches="tight")
    plt.show()

    plt.hist(HVratiozen_ice, bin_edges, alpha=0.6, edgecolor='black')
    plt.xlabel('Hpole/Vpole')
    plt.ylabel('Nant')
    #plt.xlim(0,2000)
    plt.legend()
    #if(scale=="log"): plt.yscale("log")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.axvline(x=1, color='red', linestyle='--', linewidth=2)
    #plt.title("In-air emission")
    plt.title(r"In-ice, $\theta =%.d^{\circ}$" %ZenithBins[i])
    plt.savefig("/Users/chiche/Desktop/HVratio/InIceFilteredHVratio_zen%.d.pdf" %ZenithBins[i], bbox_inches="tight")
    plt.show()
    

EtotAirAll16_5 = \
    GetAmplitudeDistribution(HVratioAirAll, EnergyAll, PosAll, 0.0316, 3116)

EtotAirAll17 = \
    GetAmplitudeDistribution(HVratioAirAll, EnergyAll, PosAll, 0.1, 3116)

EtotAirAll17_5 = \
    GetAmplitudeDistribution(HVratioAirAll, EnergyAll, PosAll, 0.316, 3116)

EtotIceAll16_5 = \
    GetAmplitudeDistribution(HVratioIceAll, EnergyAll, PosAll, 0.0316, 3116)

EtotIceAll17 = \
    GetAmplitudeDistribution(HVratioIceAll, EnergyAll, PosAll, 0.1, 3116)

EtotIceAll17_5 = \
    GetAmplitudeDistribution(HVratioIceAll, EnergyAll, PosAll, 0.316, 3116)

### Plots
labels=('$10^{16.5}$ eV', '$10^{17}$ eV', '$10^{17.5}$ eV')
bin_edges = np.linspace(0, 10, 80) 


plt.hist(EtotAirAll16_5, bin_edges, alpha=0.6, edgecolor='black', label=labels[0])
plt.hist(EtotAirAll17, bin_edges, alpha=0.6, edgecolor='black', label=labels[1])
plt.hist(EtotAirAll17_5 , bin_edges, alpha=0.6, edgecolor='black', label=labels[2])
plt.xlabel('Hpole/Vpole')
plt.ylabel('Nant')
#plt.xlim(0,2000)
plt.legend()
#if(scale=="log"): plt.yscale("log")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.axvline(x=1, color='red', linestyle='--', linewidth=2)
plt.title("In-air emission")
#plt.title(r"In-air, $\theta =0^{\circ}$, $E=10^{17.5} eV$")
plt.savefig("/Users/chiche/Desktop/InAirFilteredHVratio.pdf", bbox_inches="tight")
plt.show()

plt.hist(EtotIceAll16_5, bin_edges, alpha=0.6, edgecolor='black', label=labels[0])
plt.hist(EtotIceAll17, bin_edges, alpha=0.6, edgecolor='black', label=labels[1])
plt.hist(EtotIceAll17_5 , bin_edges, alpha=0.6, edgecolor='black', label=labels[2])
plt.xlabel('Hpole/Vpole')
plt.ylabel('Nant')
#plt.xlim(0,2000)
plt.legend()
#if(scale=="log"): plt.yscale("log")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.axvline(x=1, color='red', linestyle='--', linewidth=2)
plt.title("In-ice emission")
#plt.title(r"In-air, $\theta =0^{\circ}$, $E=10^{17.5} eV$")
plt.savefig("/Users/chiche/Desktop/InIceFilteredHVratio.pdf", bbox_inches="tight")
plt.show()

PlotAmplitudeDistribution(EtotAirAll16_5, EtotAirAll17, EtotAirAll17_5, bin_edges, labels)

bin_edges = np.linspace(0, 10, 60) 
PlotAmplitudeDistribution(EtotIceAll16_5, EtotIceAll17, EtotIceAll17_5, bin_edges, labels, "log")

