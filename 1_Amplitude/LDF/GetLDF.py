#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 02:21:35 2024

@author: chiche
"""
#SCRIPT NEEDS UPDATES!

# Modules import
#region Modules 
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
import glob
import sys
import pickle
from MainModules.ShowerClass import CreateShowerfromHDF5
from MainModules.PlotConfig import MatplotlibConfig
sys.path.append("/Users/chiche/Desktop/DeepCrSearch/Analysis/")
##from Modules.SimParam.GetLayoutScaling import GetDepthScaling, GetEnergyScaling
from scipy.interpolate import interp1d
import scipy
#from FunctionsPlotFluence import EfieldMap, PlotLDF, PlotTraces, plot_polarisation, PlotMaxTraces, PlotAllTraces, PlotLayer, PlotGivenTrace, PlotAllChannels, PlotSurfaceEz
from Modules.ModulePlotRadioSimExtent import PlotFillingFactor, PlotRadioSimExtent, PlotAirIceExtent, PlotMaxLDF
from scipy.interpolate import griddata
from datetime import datetime
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt
from Modules.ModuleSimExtent import GetRadioExtent, GetMaxLDF, GetCaracExtent
##from Modules.Fluence.FunctionsRadiationEnergy import GetFluence, GetRadiationEnergy
#endregion

#region Path definition
SimDir = "DeepCrLibV1"  #"InterpSim"
SimName = "Rectangle_Proton_0.316_0_0_1_0.hdf5"
WorkPath = os.getcwd()
simpath = "/Users/chiche/Desktop/DeepCrAnalysis/Simulations/" + SimDir + "/"\
      + SimName 
BatchID = ""
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
#                           Get peak amplitude
# =============================================================================

ExC, EyC, EzC, EtotC = Shower.GetPeakTraces(Traces_C)
ExG, EyG, EzG, EtotG = Shower.GetPeakTraces(Traces_G)
#Extot, Eytot, Eztot, Etot_peak = Shower.GetPeakTraces(Traces_tot)

# =============================================================================
#                                 Get integral
# =============================================================================

ExC_int, EyC_int, EzC_int, EtotC_int, peaktime = Shower.GetIntTraces(Traces_C)
ExG_int, EyG_int, EzG_int, EtotG_int, peaktime = Shower.GetIntTraces(Traces_G)
#Ex_tot_int, Ey_tot_int, Ez_tot_int, Etot_int = Shower.GetIntTraces(Traces_tot)

# =============================================================================
#                    Parametrization of the spacing
# =============================================================================

#radioextent, simextent, extent, maxpos, xminlay, xmaxlay = \
#    GetRadioExtent(Nlay, Nplane, Pos, Etot_int)

#airextent, simextent, extent, maxpos, xminlay, xmaxlay = \
#    GetRadioExtent(Nlay, Nplane, Pos, EtotC_int)

#icextent, simextent, extent, maxpos, xminlay, xmaxlay = \
#    GetRadioExtent(Nlay, Nplane, Pos, EtotG_int)

# =============================================================================
#                                  LDF
# =============================================================================

# Surface antennas results
posx_MaxAirLDF, Ex_MaxAirLDF = GetMaxLDF(Pos, EtotC_int, Depths[0], "x") 
posx_MaxIceLDF, Ex_MaxIceLDF = GetMaxLDF(Pos, EtotG_int, Depths[0], "x") 
Lextentx_air, Lmaxx_air = GetCaracExtent(posx_MaxAirLDF[:,0], Ex_MaxAirLDF)
Lextentx_ice, Lmaxx_ice = GetCaracExtent(posx_MaxIceLDF[:,0], Ex_MaxIceLDF)


posy_MaxAirLDF, Ey_MaxAirLDF = GetMaxLDF(Pos, EtotC_int, Depths[0], "y") 
posy_MaxIceLDF, Ey_MaxIceLDF = GetMaxLDF(Pos, EtotG_int, Depths[0], "y") 
Lextenty_air, Lmaxy_air = GetCaracExtent(posy_MaxAirLDF[:,1], Ey_MaxAirLDF)
Lextenty_ice, Lmaxy_ice = GetCaracExtent(posx_MaxIceLDF[:,1], Ey_MaxIceLDF)

# 60m-deep antennas results
posx60_MaxAirLDF, Ex60_MaxAirLDF = GetMaxLDF(Pos, EtotC_int, Depths[2], "x") 
posx60_MaxIceLDF, Ex60_MaxIceLDF = GetMaxLDF(Pos, EtotG_int, Depths[2], "x") 
Lextentx60_air, Lmaxx60_air = GetCaracExtent(posx60_MaxAirLDF[:,0], Ex60_MaxAirLDF)
Lextentx60_ice, Lmaxx60_ice = GetCaracExtent(posx60_MaxIceLDF[:,0], Ex60_MaxIceLDF)


posy60_MaxAirLDF, Ey60_MaxAirLDF = GetMaxLDF(Pos, EtotC_int, Depths[2], "y") 
posy60_MaxIceLDF, Ey60_MaxIceLDF = GetMaxLDF(Pos, EtotG_int, Depths[2], "y") 
Lextenty60_air, Lmaxy60_air = GetCaracExtent(posy60_MaxAirLDF[:,1], Ey60_MaxAirLDF)
Lextenty60_ice, Lmaxy60_ice = GetCaracExtent(posy60_MaxIceLDF[:,1], Ey60_MaxIceLDF)


# 100m-deep antennas results
posx100_MaxAirLDF, Ex100_MaxAirLDF = GetMaxLDF(Pos, EtotC_int, Depths[4], "x") 
posx100_MaxIceLDF, Ex100_MaxIceLDF = GetMaxLDF(Pos, EtotG_int, Depths[4], "x") 
Lextentx100_air, Lmaxx100_air = GetCaracExtent(posx100_MaxAirLDF[:,0], Ex100_MaxAirLDF)
Lextentx100_ice, Lmaxx100_ice = GetCaracExtent(posx100_MaxIceLDF[:,0], Ex100_MaxIceLDF)


posy100_MaxAirLDF, Ey100_MaxAirLDF = GetMaxLDF(Pos, EtotC_int, Depths[4], "y") 
posy100_MaxIceLDF, Ey100_MaxIceLDF = GetMaxLDF(Pos, EtotG_int, Depths[4], "y") 
Lextenty100_air, Lmaxy100_air = GetCaracExtent(posy100_MaxAirLDF[:,1], Ey100_MaxAirLDF)
Lextenty100_ice, Lmaxy100_ice = GetCaracExtent(posy100_MaxIceLDF[:,1], Ey100_MaxIceLDF)


# =============================================================================
#                              PLOTS
# =============================================================================
#PlotAirIceExtent(Depths, airextent, icextent, simextent, energy, theta)
#PlotRadioSimExtent(Depths, radioextent, simextent)
#PlotFillingFactor(Depths, radioextent, simextent)
#path = "/Users/chiche/Desktop/RNOG_Brussels_10_04_25/Plots/LDFs"
PlotMaxLDF(posx_MaxAirLDF, posx_MaxIceLDF, Ex_MaxAirLDF, Ex_MaxIceLDF, "x", Depths[0], Shower, OutputPath)
PlotMaxLDF(posy_MaxAirLDF, posy_MaxIceLDF, Ey_MaxAirLDF, Ey_MaxIceLDF, "y",Depths[0], Shower, OutputPath)
PlotMaxLDF(posx60_MaxAirLDF, posx60_MaxIceLDF, Ex60_MaxAirLDF, Ex60_MaxIceLDF, "x", Depths[2], Shower, OutputPath)
PlotMaxLDF(posy60_MaxAirLDF, posy60_MaxIceLDF, Ey60_MaxAirLDF, Ey60_MaxIceLDF, "y",Depths[2], Shower, OutputPath)
PlotMaxLDF(posx100_MaxAirLDF, posx100_MaxIceLDF, Ex100_MaxAirLDF, Ex100_MaxIceLDF, "x", Depths[4], Shower, OutputPath)
PlotMaxLDF(posy100_MaxAirLDF, posy100_MaxIceLDF, Ey100_MaxAirLDF, Ey100_MaxIceLDF, "y",Depths[4], Shower, OutputPath)

#np.savetxt("ProtonLDFdataPosIce100.txt", posx100_MaxIceLDF)
#np.savetxt("ProtonLDFdataEIce100.txt",Ex100_MaxIceLDF )
