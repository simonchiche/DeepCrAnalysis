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

    Filter = False
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
                        [EtotC_int, EtotG_int, energy, theta, Pos]):
        lst.append(val)


EtotAirAll, EtotIceAll, EnergyAll, ZenithAll, PosAll =\
      map(np.array, [EtotAirAll, EtotIceAll, EnergyAll, ZenithAll, PosAll])



EtotAirAll16_5 = \
    GetAmplitudeDistribution(EtotAirAll, EnergyAll, PosAll, 0.0316, 3116)

EtotAirAll17 = \
    GetAmplitudeDistribution(EtotAirAll, EnergyAll, PosAll, 0.1, 3116)

EtotAirAll17_5 = \
    GetAmplitudeDistribution(EtotAirAll, EnergyAll, PosAll, 0.316, 3116)

EtotIceAll16_5 = \
    GetAmplitudeDistribution(EtotIceAll, EnergyAll, PosAll, 0.0316, 3116)

EtotIceAll17 = \
    GetAmplitudeDistribution(EtotIceAll, EnergyAll, PosAll, 0.1, 3116)

EtotIceAll17_5 = \
    GetAmplitudeDistribution(EtotIceAll, EnergyAll, PosAll, 0.316, 3116)

### Plots
labels=('$10^{16.5}$ eV', '$10^{17}$ eV', '$10^{17.5}$ eV')
bin_edges = np.linspace(0, 7000, 80) 
PlotAmplitudeDistribution(EtotAirAll16_5, EtotAirAll17, EtotAirAll17_5, bin_edges, labels)

bin_edges = np.linspace(0, 6000, 60) 
PlotAmplitudeDistribution(EtotIceAll16_5, EtotIceAll17, EtotIceAll17_5, bin_edges, labels, "log")

