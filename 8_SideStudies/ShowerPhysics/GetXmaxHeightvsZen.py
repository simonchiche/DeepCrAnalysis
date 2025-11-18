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
from ModulePlotXmax import plot_Xmax_pos, PlotXmaxHeightvsZenith, PlotXmaxGrammage, GenerateXmaxIceData, PlotXmaxDistribution, PlotSlantXmaxIce, PlotXmaxIceDepth, PlotIceXmaxDistribution
##from Modules.SimParam.GetLayoutScaling import GetDepthScaling, GetEnergyScaling
from scipy.interpolate import interp1d
import scipy
##from Modules.Fluence.FunctionsGetFluence import  Norm, LoadTraces, GetPeakTraces, Traces_cgs_to_si, GetDepths, CorrectScaling, CombineTraces, CorrectLength, GetIntTraces, GetIntTracesSum, GetRadioExtent
##from CleanCoreasTraces import CleanCoreasTraces
##from Modules.SimParam.PlotRadioSimExtent import PlotFillingFactor, PlotRadioSimExtent, PlotAirIceExtent
from scipy.interpolate import griddata
from datetime import datetime
from scipy.optimize import curve_fit
from ModuleGetAirXmaxPos import getXmaxPosition
##from Modules.Fluence.FunctionsRadiationEnergy import GetFluence, GetRadiationEnergy
#endregion

#region Path definition
SimDir = "FullDenseDeepCr" #"DeepCrLibV1"  #"InterpSim"
WorkPath = os.getcwd()
BatchID = "XmaxPos"
OutputPath = MatplotlibConfig(WorkPath, SimDir, BatchID)
#endregion
Save = True
simpath = "/Users/chiche/Desktop/DeepCrAnalysis/Simulations/" + SimDir
SimpathAll = glob.glob(simpath + "/*")

EnergyAll = []
ZenithAll = []
XmaxPosAll = []
XmaxAll = []

for simpath in SimpathAll:
    print(simpath.split("/")[-1])
    Shower = CreateShowerfromHDF5(simpath)

# =============================================================================
#                              Load Traces
# =============================================================================

    energy, theta, azimuth, injection, glevel, xmaxairdist,  Xmax = \
        Shower.energy, Shower.zenith, Shower.azimuth, Shower.injection, Shower.glevel,  Shower.xmaxdist/1e2, Shower.xmax
    Traces_C, Traces_G, Pos = Shower.traces_c, Shower.traces_g, Shower.pos
    EnergyAll.append(energy)
    ZenithAll.append(theta)
    print(energy, theta, azimuth, injection, glevel, xmaxairdist,  Xmax)

    Xmaxpos_air = getXmaxPosition(0, theta, glevel, injection, xmaxairdist)
    XmaxPosAll.append(Xmaxpos_air)
    XmaxAll.append(Xmax)

EnergyAll, ZenithAll, XmaxPosAll, XmaxAll = \
    np.array(EnergyAll), np.array(ZenithAll), np.array(XmaxPosAll), np.array(XmaxAll)

# =============================================================================
#                             Air Xmax
# =============================================================================

plot_Xmax_pos(EnergyAll, XmaxPosAll, OutputPath)

PlotXmaxHeightvsZenith(EnergyAll, ZenithAll, XmaxPosAll, OutputPath)


PlotXmaxHeightvsZenith(EnergyAll, ZenithAll, XmaxPosAll, OutputPath)

PlotXmaxGrammage(EnergyAll, ZenithAll, XmaxAll, OutputPath)


PlotXmaxDistribution(EnergyAll, XmaxAll, OutputPath)

# =============================================================================
#                             Ice Xmax
# =============================================================================

### Create Ice Xmax Data
DataPath = "./XmaxIce/*.long"
GenerateXmaxIceData(DataPath)

XmaxIceData = np.loadtxt("./XmaxIce/XmaxIceData.txt")
XmaxIceAll, EiceAll, ZenIceAll =\
      XmaxIceData[:,0], XmaxIceData[:,1], XmaxIceData[:,2]


PlotIceXmaxDistribution(EiceAll, XmaxIceAll, OutputPath)

PlotSlantXmaxIce(EiceAll, ZenIceAll, XmaxIceAll, OutputPath)

PlotXmaxIceDepth(EiceAll, ZenIceAll, XmaxIceAll, OutputPath)


