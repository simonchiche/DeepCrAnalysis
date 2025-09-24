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
from Modules.ModuleGetAmplitudeDistrib import GetAmplitudeDistribution, GetAmplitudeDistributionZenBin
##from Modules.Fluence.FunctionsRadiationEnergy import GetFluence, GetRadiationEnergy
#endregion

#region Path definition
SimDir = "FullDenseDeepCr" #"DeepCrLibV1"  #"InterpSim"
WorkPath = os.getcwd()
BatchID = "AmplitudeDistrib_filtered"
OutputPath = MatplotlibConfig(WorkPath, SimDir, BatchID)
print("OutputPath: ", OutputPath)
#endregion
Save = False
simpath = "/Users/chiche/Desktop/DeepCrAnalysis/Simulations/" + SimDir
SimpathAll = glob.glob(simpath + "/*")

EnergyAll, ZenithAll, EtotAirAll, EtotIceAll, PosAll = \
    ([] for _ in range(5))

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
    for lst, val in zip([EtotAirAll, EtotIceAll, EnergyAll, ZenithAll, PosAll],
                        [EtotC, EtotG, energy, theta, Pos]):
        lst.append(val)


EtotAirAll, EtotIceAll, EnergyAll, ZenithAll, PosAll =\
      map(np.array, [EtotAirAll, EtotIceAll, EnergyAll, ZenithAll, PosAll])


EnergyBins = np.sort(np.unique(EnergyAll))
ZenithBins = np.sort(np.unique(ZenithAll))

zenithbins = [0, 34, 50]
EtotAirAll16_5_zen, EtotAirAll17_5_zen = {}, {}
for zen in zenithbins:
    EtotAirAll16_5_zen[zen] = \
        GetAmplitudeDistributionZenBin(EtotAirAll, EnergyAll, ZenithAll, PosAll, EnergyBins[0], zen, 3116)
    EtotAirAll17_5_zen[zen] = \
        GetAmplitudeDistributionZenBin(EtotAirAll, EnergyAll, ZenithAll, PosAll, EnergyBins[1], zen, 3116)


EtotAirAll16_5 = \
    GetAmplitudeDistribution(EtotAirAll, EnergyAll, PosAll, EnergyBins[0], 3116)

#EtotAirAll17 = \
#    GetAmplitudeDistribution(EtotAirAll, EnergyAll, PosAll, EnergyBins[1], 3116)

#EtotAirAll17_5 = \
#    GetAmplitudeDistribution(EtotAirAll, EnergyAll, PosAll, EnergyBins[2], 3116)
EtotAirAll17_5 = \
    GetAmplitudeDistribution(EtotAirAll, EnergyAll, PosAll, EnergyBins[1], 3116)


EtotIceAll16_5 = \
    GetAmplitudeDistribution(EtotIceAll, EnergyAll, PosAll, EnergyBins[0], 3116)

#EtotIceAll17 = \
#    GetAmplitudeDistribution(EtotIceAll, EnergyAll, PosAll, EnergyBins[1], 3116)

#EtotIceAll17_5 = \
#    GetAmplitudeDistribution(EtotIceAll, EnergyAll, PosAll, EnergyBins[2], 3116)
EtotIceAll17_5 = \
    GetAmplitudeDistribution(EtotIceAll, EnergyAll, PosAll, EnergyBins[1], 3116)


### Plots
labels=('$10^{16.5}$ eV', '$10^{17}$ eV', '$10^{17.5}$ eV')

#bin_edges = np.linspace(20,6000, 50) 
#PlotAmplitudeDistribution(EtotAirAll16_5, EtotAirAll17, EtotAirAll17_5, bin_edges, labels, True, OutputPath, "In-air", "log")

#bin_edges = np.linspace(20, 6000, 50) 
#PlotAmplitudeDistribution(EtotIceAll16_5, EtotIceAll17, EtotIceAll17_5, bin_edges, labels, True, OutputPath, "In-ice", "log")

def PlotAmplitudeDistribution(Etot1, Etot3, bin_edges, labels, Save, OutputPath, pretitle, scale = "linear"):


    plt.hist(Etot1, bin_edges, alpha=0.6, edgecolor='black', label=labels[0])
    plt.hist(Etot3 , bin_edges, alpha=0.6, edgecolor='black', label=labels[2])
    plt.xlabel('$E_{tot}^{peak}\, [\mu V /m]$')
    plt.ylabel('Nant')
    #plt.xlim(0,2000)
    plt.legend()
    if(scale=="log"): plt.yscale("log")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    #plt.title(r"In-air, $\theta =0^{\circ}$, $E=10^{17.5} eV$")]
    OutputPath = OutputPath + pretitle
    #plt.xscale("log")
    plt.title(f"{pretitle} emission")
    plt.tight_layout()
    if(Save):
        plt.savefig(OutputPath + "FilteredPulseDistrib.pdf", bbox_inches="tight") 
    plt.show()

bin_edges = np.linspace(20, 6000, 50) 
PlotAmplitudeDistribution(EtotAirAll16_5, EtotAirAll17_5, bin_edges, labels, True, OutputPath, "In-air", "lin")
