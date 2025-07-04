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
##from Modules.Fluence.FunctionsRadiationEnergy import GetFluence, GetRadiationEnergy
#endregion

#region Path definition
SimDir = "DeepCrLibV1"  #"InterpSim"
SimName = "Rectangle_Proton_0.0316_0_0_1_0.hdf5"
WorkPath = os.getcwd()
simpath = "/Users/chiche/Desktop/DeepCrAnalysis/Simulations/"\
+ SimDir + "/" + SimName 
BatchID = "Proton_vs_Gamma"
OutputPath = MatplotlibConfig(WorkPath, SimDir, BatchID)
#endregion
Save = False
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


# =============================================================================
#                            Cleaning the data
# =============================================================================

#region remove core antennas
#Pos, EtotC_int, Ex_tot_int, Ey_tot_int, Ez_tot_int \
#    = RemoveCoreAntennas(Pos, 15, Ex_tot_int, Ey_tot_int, Ez_tot_int, EtotC_int)
#endregion

# =============================================================================
#                         Compute Fluence
# =============================================================================

# Coreas
EfieldMap(Pos, Depths, Nplane, np.log10(EtotC_int), "In-air", \
          Save, energy, theta, OutputPath)

# Geant 
EfieldMap(Pos, Depths, Nplane, np.log10(EtotG_int+ 1), "In-ice_proton",\
          Save, energy, theta, OutputPath)
    
# z-component
#EfieldMap(Pos, Depths, Nplane, np.log10(EzC_int), "Log10(Ez) CoreasHilbert", \
#          False, energy, theta, OutputPath)

# Coreas Normalized
#EfieldMap(Pos, Nlay, Nplane, EtotC_int/max(EtotC_int), "Coreas",\
#          False, energy, theta, OutputPath)


# Geant normalized
##EfieldMap(Pos, Depths, Nplane, EtotG_int/max(EtotG_int), "Geant", \
##          False, energy, theta, OutputPath)

# Total emission
#EfieldMap(Pos, Depths, Nplane, Etot_int, "Total", \
#          False, energy, theta, OutputPath)

#Total emission from peak
#EfieldMap(Pos, Depths, Nplane, np.maximum(EtotC, EtotG), "Total",\
#          False, energy, theta, OutputPath)

# Geant over CoREAS
##EfieldMap(Pos, Depths, Nplane, EtotG_int/EtotC_int, "GeantoverCoreas",\
##          False, energy, theta, OutputPath)

