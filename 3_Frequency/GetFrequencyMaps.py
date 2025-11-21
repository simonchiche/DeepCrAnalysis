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
from Modules.ModuleFrequencySpectrum import compute_spectrum, PlotAllSpectra, PlotFrequencyHeatmap, PlotAllSignals, GetPeakTraces, PlotAllSpectra_rbin
##from CleanCoreasTraces import CleanCoreasTraces
##from Modules.SimParam.PlotRadioSimExtent import PlotFillingFactor, PlotRadioSimExtent, PlotAirIceExtent
from scipy.interpolate import griddata
from datetime import datetime
from scipy.optimize import curve_fit
##from Modules.Fluence.FunctionsRadiationEnergy import GetFluence, GetRadiationEnergy
#endregion

#region Path definition
SimDir = "FullDenseDeepCr" #"DeepCrLibV1"  #"InterpSim"
SimName = "Polar_Proton_0.316_0_0_1.hdf5" #"Rectangle_Proton_0.0316_0_0_1_0.hdf5"
WorkPath = os.getcwd()
simpath = "/Users/chiche/Desktop/DeepCrAnalysis/Simulations/"\
+ SimDir + "/" + SimName 
BatchID = "FrequencySpectrum"
OutputPath = MatplotlibConfig(WorkPath, SimDir, BatchID)
#endregion
Save = True
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

DeepPos = Pos[Pos[:,2]==Depths[2]]
selDeep = Pos[:,2]==Depths[2]
Trace_C_Deep = {k: v for (k, v), m in zip(Traces_C.items(), selDeep) if m}
Trace_C_Deep = {i: v for i, v in enumerate(Trace_C_Deep.values())}
Trace_G_Deep = {k: v for (k, v), m in zip(Traces_G.items(), selDeep) if m}
Trace_G_Deep = {i: v for i, v in enumerate(Trace_G_Deep.values())}

PlotAllSpectra(Trace_C_Deep)
PlotAllSpectra(Trace_G_Deep)

radius = np.sqrt(DeepPos[:,0]**2 + DeepPos[:,1]**2) 
radius_idx = np.argsort(radius)

### Frequency spectra per radius bin ###
radius_bins = np.linspace(0, 740, 50)
for i in range(len(radius_bins)-1):
    mask_rad = (radius >= radius_bins[i]) & (radius < radius_bins[i+1])
    Trace_G_Deep_bin = {k: v for (k, v), m in zip(Trace_G_Deep.items(), mask_rad) if m}
    Trace_G_Deep_bin = {j: v for j, v in enumerate(Trace_G_Deep_bin.values())}
    print(f"Radius bin: {radius_bins[i]} - {radius_bins[i+1]} m, N_antennas: {len(Trace_G_Deep_bin)}")
    if len(Trace_G_Deep_bin) > 0:
        PlotAllSpectra(Trace_G_Deep_bin)
        #PlotAllSignals(Trace_G_Deep_bin)

radius_bins = np.linspace(0, 740, 50)
for i in range(len(radius_bins)-1):
    mask_rad = (radius >= radius_bins[i]) & (radius < radius_bins[i+1])
    Trace_C_Deep_bin = {k: v for (k, v), m in zip(Trace_C_Deep.items(), mask_rad) if m}
    Trace_C_Deep_bin = {j: v for j, v in enumerate(Trace_C_Deep_bin.values())}
    print(f"Radius bin: {radius_bins[i]} - {radius_bins[i+1]} m, N_antennas: {len(Trace_G_Deep_bin)}")
    if len(Trace_C_Deep_bin) > 0:
        PlotAllSpectra(Trace_C_Deep_bin)
        #PlotAllSignals(Trace_C_Deep_bin)


## In-air spectra on the Cerenkov cone
radius_bins = np.linspace(0, 740, 50)
mask_rad = (radius >= 0) & (radius <20)
Trace_C_Deep_bin = {k: v for (k, v), m in zip(Trace_C_Deep.items(), mask_rad) if m}
Trace_C_Deep_bin = {j: v for j, v in enumerate(Trace_C_Deep_bin.values())}
print(f"Radius bin: {radius_bins[i]} - {radius_bins[i+1]} m, N_antennas: {len(Trace_G_Deep_bin)}")
if len(Trace_C_Deep_bin) > 0:
    PlotAllSpectra_rbin(Trace_C_Deep_bin, "In-air", OutputPath)

## In-ice spectra on the Cerenkov cone
mask_rad = (radius >= 60) & (radius <75)
Trace_G_Deep_bin = {k: v for (k, v), m in zip(Trace_G_Deep.items(), mask_rad) if m}
Trace_G_Deep_bin = {j: v for j, v in enumerate(Trace_G_Deep_bin.values())}
print(f"Radius bin: {radius_bins[i]} - {radius_bins[i+1]} m, N_antennas: {len(Trace_G_Deep_bin)}")
if len(Trace_G_Deep_bin) > 0:
    PlotAllSpectra_rbin(Trace_G_Deep_bin,"In-ice", OutputPath)
    #PlotAllSignals(Trace_G_Deep_bin)
#################

radius = np.sqrt(DeepPos[:,0]**2 + DeepPos[:,1]**2) 
radius_idx = np.argsort(radius)

Trigger = False
Threshold = 0
if(Trigger):    
    Ex, Ey, Ez, Etot_C = GetPeakTraces(Trace_C_Deep)
    selE = Etot_C>Threshold
    Trace_C_Deep_highE = {k: v for (k, v), m in zip(Trace_G_Deep.items(), selE) if m}
    Trace_C_Deep_highE = {i: v for i, v in enumerate(Trace_C_Deep_highE.values())}
    radius_highE = radius[selE]
    radius_idx = np.argsort(radius_highE)

PlotFrequencyHeatmap(Trace_C_Deep, radius, radius_idx, Shower, OutputPath, label="In-air", rmax=250, Save=False, merge_factor=1)

PlotFrequencyHeatmap(Trace_G_Deep, radius, radius_idx, Shower, OutputPath, label="In-ice", rmax=250, Save=False, merge_factor=1)





