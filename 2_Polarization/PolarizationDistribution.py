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
from Modules.ModulePlotPolarization import PlotHVratioAirDistribperZen, PlotHVratioIceDistribperZen, GetHVratioAirvsE, GetHVratioIcevsE
##from Modules.Fluence.FunctionsRadiationEnergy import GetFluence, GetRadiationEnergy
#endregion

#region Path definition
SimDir = "DeepCrLib"  #"InterpSim"
WorkPath = os.getcwd()
BatchID = "PolarizationDistribution"
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
    triggerthresold=0
    selair = EtotC_int > triggerthresold
    selice = EtotG_int > triggerthresold
    ExC_int, EyC_int, EzC_int = np.array(ExC_int)[selair], np.array(EyC_int)[selair], np.array(EzC_int)[selair]
    ExG_int, EyG_int, EzG_int = np.array(ExG_int)[selice], np.array(EyG_int)[selice], np.array(EzG_int)[selice]
    HVratioAir = np.sqrt((np.array(ExC_int)**2 + np.array(EyC_int)**2))/(np.array(EzC_int))
    HVratioIce = np.sqrt((np.array(ExG_int)**2 + np.array(EyG_int)**2))/(np.array(EzG_int))

    HVratioAir[~np.isnan(HVratioAir)]
    HVratioIce[~np.isnan(HVratioIce)]
    for lst, val in zip([HVratioAirAll, HVratioIceAll, ExAirAll, EyAirAll, EzAirAll,  ExIceAll, EyIceAll, EzIceAll, EnergyAll, ZenithAll, PosAll],
                        [HVratioAir, HVratioIce, ExC_int, EyC_int, EzC_int, ExG_int, EyG_int, EzG_int, energy, theta, Pos]):
        lst.append(val)

bin_edges = np.linspace(0, 10, 80) 
plt.hist(HVratioAirAll[0], bins=bin_edges)
plt.hist(HVratioIceAll[0], bins=bin_edges)
plt.show()
HVratioIceAll = np.shape(HVratioIceAll)
print(len(HVratioAirAll[15]))

# Getting the Air/Ice Hpol/Vpol ratio at 100 m depth, for each energy bin
selDepth =3116
selE = 0.0316
HVratioAirAll16_5 = \
    GetAmplitudeDistribution(HVratioAirAll, EnergyAll, PosAll, 0.0316, 3116)
HVratioIceAll16_5 = \
    GetAmplitudeDistribution(HVratioIceAll, EnergyAll, PosAll, 0.0316, 3116)

selE = 0.1
HVratioAirAll17 = \
    GetAmplitudeDistribution(HVratioAirAll, EnergyAll, PosAll, 0.1, 3116)
HVratioIceAll17 = \
    GetAmplitudeDistribution(HVratioIceAll, EnergyAll, PosAll, 0.1, 3116)

selE = 0.316
def GetAmplitudeDistribution(Etot, EnergyAll, PosAll, SelE, SelDepth):

    MaskE = (EnergyAll == SelE)
    #MaskDepth = (PosAll[MaskE].reshape(-1, 3)[:,2] == SelDepth)
    Emasked = Etot[MaskE]

    EmaskedDepth = []
    for i in range(len(Emasked)):

        MaskDepth = PosAll[i][:,2] == SelDepth
        EmaskedDepth.append(Emasked[i])#[MaskDepth])
    
    return np.array(EmaskedDepth).flatten()
HVratioAirAll17_5 = \
    GetAmplitudeDistribution(HVratioAirAll, EnergyAll, PosAll, 0.316, 3116)
HVratioIceAll17_5 = \
    GetAmplitudeDistribution(HVratioIceAll, EnergyAll, PosAll, 0.316, 3116)


# HV ratio  for each zenith

def PlotHVratioIceDistribperZen(ZenithAll, HVratioIceAll, Save, OutputPath):
    
    ZenithBins = np.unique(ZenithAll)
    bin_edges = np.linspace(0, 10, 80) 
    for i in range(len(ZenithBins)):
        
        sel = (ZenithAll == ZenithBins[i])    

        HVratiozen_ice = HVratioIceAll[sel].flatten()

        plt.hist(HVratiozen_ice, bin_edges, alpha=0.6, edgecolor='black')
        plt.xlabel('Hpol/Vpol')
        plt.ylabel('Nant')
        #plt.xlim(0,2000)
        plt.legend()
        #if(scale=="log"): plt.yscale("log")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.axvline(x=np.sqrt(2), color='red', linestyle='--', linewidth=2)
        #plt.title("In-air emission")
        plt.title(r"In-ice, $\theta =%.d^{\circ}$" %ZenithBins[i])
        #plt.savefig(OutputPath + "InIceFilteredHVratio_zen%.d.pdf" %ZenithBins[i], bbox_inches="tight") if Save else None
        plt.show()
# In-air
PlotHVratioAirDistribperZen(ZenithAll, HVratioAirAll, Save, OutputPath)
# In-ice
PlotHVratioIceDistribperZen(ZenithAll, HVratioIceAll, Save, OutputPath)

Save= True
bin_edges = np.linspace(0, 10, 80) 
plt.hist(HVratioIceAll17_5, bin_edges, alpha=0.6, edgecolor='black', label="In-ice", color="#4AB8E7")
plt.hist(HVratioAirAll17_5, bin_edges, alpha=0.6, edgecolor='black', label="In-air", color="#E74A6B")
plt.xlabel('Hpol/Vpol')
plt.ylabel('Nant')
#plt.xlim(0,2000)
plt.legend()
#if(scale=="log"): plt.yscale("log")
plt.grid(axis='y', linestyle='--', alpha=0.7)
#plt.axvline(x=np.sqrt(2), color='red', linestyle='--', linewidth=2)
#plt.title("In-air emission")
plt.title(r"All zeniths, $E=10^{17.5}$ eV", fontsize=14)
plt.legend()
plt.savefig(OutputPath + "AirvsIce.pdf", bbox_inches="tight") if Save else None
plt.show()

plt.hist(HVratioAirAll17_5, bin_edges, density=True)
plt.hist(HVratioIceAll17_5, bin_edges, density=True)
#HV ratio distrib, all zenith: energy bins comparison
GetHVratioAirvsE(HVratioAirAll16_5, HVratioAirAll17, HVratioAirAll17_5, OutputPath)
GetHVratioIcevsE(HVratioIceAll16_5, HVratioIceAll17, HVratioIceAll17_5, OutputPath)

bin_edges = np.linspace(0, 10, 80) 
PlotAmplitudeDistribution(HVratioAirAll16_5, HVratioAirAll17, HVratioAirAll17_5, bin_edges)

bin_edges = np.linspace(0, 10, 60) 
PlotAmplitudeDistribution(HVratioIceAll16_5, HVratioIceAll17, HVratioIceAll17_5, bin_edges)


HVratioIceAll17_5_cleaned = HVratioIceAll17_5[~np.isnan(HVratioIceAll17_5)]
print(len(HVratioIceAll17_5_cleaned))
