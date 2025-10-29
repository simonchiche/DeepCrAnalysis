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

Ebins = np.unique(EnergyAll)

maskE = EnergyAll == Ebins[1]

for i in range(len(Ebins)):
    maskE = EnergyAll == Ebins[i]
# Xmax Position for different zenith
    plt.scatter(XmaxPosAll[:,0][maskE], XmaxPosAll[:,2][maskE], label = f"E = {Ebins[i]} EeV")
    plt.xlabel("x [m]")
    plt.ylabel("Xmax height [m]")
    plt.legend()
    plt.grid()
    plt.ylim(2000, max(XmaxPosAll[:,2])+200)
    plt.axhspan(2000, 3216, color='skyblue', alpha=0.3) 
plt.savefig(OutputPath + "XmaxAirPositions.pdf", bbox_inches = 'tight')
plt.show()

for i in range(len(Ebins)):
    maskE = EnergyAll == Ebins[i]
    plt.scatter(ZenithAll[maskE], XmaxPosAll[:,2][maskE],  label = f"E = {Ebins[i]} EeV")
    plt.xlabel("Zenith [Deg.]")
    plt.ylabel("Xmax height [m]")
    plt.legend()
    plt.ylim(2000, max(XmaxPosAll[:,2])+200)
    plt.axhspan(2000, 3216, color='skyblue', alpha=0.3) 
    plt.grid()
plt.savefig(OutputPath + "XmaxAirHeightvsZenith.pdf", bbox_inches = 'tight')
plt.show()

for i in range(len(Ebins)):
    maskE = EnergyAll == Ebins[i]
    plt.scatter(ZenithAll[maskE], XmaxAll[maskE],  label = f"E = {Ebins[i]} EeV")
    plt.xlabel("Zenith [Deg.]")
    plt.ylabel(r"Xmax Depth [$\mathrm{g/cm^2}$]")
    plt.legend()
    plt.grid()
plt.savefig(OutputPath + "XmaxAirDepth.pdf", bbox_inches = 'tight')
plt.show()

#### ICE XMAX

XmaxIceData = np.loadtxt("./XmaxIce/XmaxIceData.txt")
XmaxIceAll, EiceAll, ZenIceAll =\
      XmaxIceData[:,0], XmaxIceData[:,1], XmaxIceData[:,2]

Ebins = np.unique(EiceAll)
mask = EiceAll == Ebins[0]

for i in range(len(Ebins)):
    mask = EiceAll == Ebins[i]
    
    # Slant Xmax vs zenith
    plt.plot(ZenIceAll[mask], XmaxIceAll[mask], 'o', label = f"E = {Ebins[i]} EeV")
    plt.xlabel("Zenith [Deg.]")
    plt.ylabel("Slant ice Xmax [m]")
    plt.legend()
    plt.grid()
plt.savefig(OutputPath + "XmaxIceSlantvsZenith.pdf", bbox_inches = 'tight')
plt.show()

for i in range(len(Ebins)):
    mask = EiceAll == Ebins[i]
    
    # Xmax depth vs zenith
    plt.plot(ZenIceAll[mask], XmaxIceAll[mask]*np.cos(ZenIceAll[mask]*np.pi/180.0), 'o', label = f"E = {Ebins[i]} EeV")
    plt.xlabel("Zenith [Deg.]")
    plt.ylabel("Xmax Depth [m]")
    plt.legend()
    plt.grid()
plt.savefig(OutputPath + "XmaxIceDepthvsZenith.pdf", bbox_inches = 'tight')
plt.show()




#Data = glob.glob("./XmaxIce/*.long")
#XmaxIceAll = []
#EiceAll = []
#ZenIceAll = []
#for i in range(len(Data)):
#    XmaxIce = np.loadtxt(Data[i])
#    XmaxIceAll.append(XmaxIce)
#    EiceAll.append(float(Data[i].split("/")[-1].split("_")[1]))
#    ZenIceAll.append(float(Data[i].split("/")[-1].split("_")[2][:-4]))
#
#XmaxIceData = np.array([XmaxIceAll, EiceAll, ZenIceAll]).T
#np.savetxt("./XmaxIce/XmaxIceData.txt", XmaxIceData)