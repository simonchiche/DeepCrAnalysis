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
BatchID = "GroundParticleEnergy"
OutputPath = MatplotlibConfig(WorkPath, SimDir, BatchID)
#endregion
Save = True
simpath = "/Users/chiche/Desktop/DeepCrAnalysis/Simulations/" + SimDir
SimpathAll = glob.glob(simpath + "/*")

EnergyAll = []
ZenithAll = []
XmaxPosAll = []
XmaxAll = []
XmaxDistAll = []

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
    XmaxDistAll.append(xmaxairdist)
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
#plt.savefig(OutputPath + "XmaxAirPositions.pdf", bbox_inches = 'tight')
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
#plt.savefig(OutputPath + "XmaxAirHeightvsZenith.pdf", bbox_inches = 'tight')
plt.show()

for i in range(len(Ebins)):
    maskE = EnergyAll == Ebins[i]
    plt.scatter(ZenithAll[maskE], XmaxAll[maskE],  label = f"E = {Ebins[i]} EeV")
    plt.xlabel("Zenith [Deg.]")
    plt.ylabel(r"Xmax Depth [$\mathrm{g/cm^2}$]")
    plt.legend()
    plt.grid()
#plt.savefig(OutputPath + "XmaxAirDepth.pdf", bbox_inches = 'tight')
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
#plt.savefig(OutputPath + "XmaxIceSlantvsZenith.pdf", bbox_inches = 'tight')
plt.show()

for i in range(len(Ebins)):
    mask = EiceAll == Ebins[i]
    
    # Xmax depth vs zenith
    plt.plot(ZenIceAll[mask], XmaxIceAll[mask]*np.cos(ZenIceAll[mask]*np.pi/180.0), 'o', label = f"E = {Ebins[i]} EeV")
    plt.xlabel("Zenith [Deg.]")
    plt.ylabel("Xmax Depth [m]")
    plt.legend()
    plt.grid()
#plt.savefig(OutputPath + "XmaxIceDepthvsZenith.pdf", bbox_inches = 'tight')
plt.show()

plt.scatter(EnergyAll, XmaxDistAll)
plt.show()
sys.exit()



import glob
DataAll = glob.glob("./Data/GroundParticleFiles/*")

Etot = []
ZenithAllpart =[]
EnergyAllpart =[]
for i in range(len(DataAll)):
    filename = DataAll[i].split("/")[-1]
    energy = float(filename.split("_")[2])
    zenith = float(filename.split("_")[3])
    EnergyAllpart.append(energy)
    ZenithAllpart.append(zenith)
    part_id, px, py, pz, x, y, t, w, E_, r_  =np.loadtxt(DataAll[i], unpack=True)
    Etot.append(np.sum(E_*w))

Etot = np.array(Etot)
EnergyAllpart = np.array(EnergyAllpart)
ZenithAllpart = np.array(ZenithAllpart)

Ebins = np.unique(EnergyAllpart)
Etot_normed = Etot/EnergyAllpart
plt.scatter(ZenithAllpart[EnergyAllpart==Ebins[0]], Etot[EnergyAllpart==Ebins[0]]/Ebins[0], label=f'E={Ebins[0]} EeV')
plt.scatter(ZenithAllpart[EnergyAllpart==Ebins[1]], Etot[EnergyAllpart==Ebins[1]]/Ebins[1], label=f'E={Ebins[1]} EeV')
plt.scatter(ZenithAllpart[EnergyAllpart==Ebins[2]], Etot[EnergyAllpart==Ebins[2]]/Ebins[2], label=f'E={Ebins[2]} EeV')
plt.legend()
plt.xlabel('Zenith Angle [Deg.]')
plt.ylabel('Normalized Total Energy at Ground')
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


XmaxDistAll = np.array(XmaxDistAll)
EnergyAll = np.array(EnergyAll)

plt.scatter(EnergyAll, XmaxDistAll)

Epart16_5=  Etot[EnergyAllpart==Ebins[0]]/Ebins[0]
Epart17=  Etot[EnergyAllpart==Ebins[1]]/Ebins[1]
Epart17_5=  Etot[EnergyAllpart==Ebins[2]]/Ebins[2]
Zenith16_5 = ZenithAllpart[EnergyAllpart==Ebins[0]]
Zenith17 = ZenithAllpart[EnergyAllpart==Ebins[1]]
Zenith17_5 = ZenithAllpart[EnergyAllpart==Ebins[2]]
arg16_5 = np.argsort(Zenith16_5)
arg17 = np.argsort(Zenith17)
arg17_5 = np.argsort(Zenith17_5)
Epart16_5 = Epart16_5[arg16_5]
Epart17 = Epart17[arg17]
Epart17_5 = Epart17_5[arg17_5]
Zenith16_5 = Zenith16_5[arg16_5]
Zenith17 = Zenith17[arg17]
Zenith17_5 = Zenith17_5[arg17_5]

XmaxDist16_5 = XmaxDistAll[EnergyAll==Ebins[0]]
XmaxDist17 = XmaxDistAll[EnergyAll==Ebins[1]]
XmaxDist17_5 = XmaxDistAll[EnergyAll==Ebins[2]]
Zenith16_5_xmax = ZenithAll[EnergyAll==Ebins[0]]
Zenith17_xmax = ZenithAll[EnergyAll==Ebins[1]]
Zenith17_5_xmax = ZenithAll[EnergyAll==Ebins[2]]
arg16_5_xmax = np.argsort(Zenith16_5_xmax)
arg17_xmax = np.argsort(Zenith17_xmax)
arg17_5_xmax = np.argsort(Zenith17_5_xmax)
XmaxDist16_5 = XmaxDist16_5[arg16_5_xmax]
XmaxDist17 = XmaxDist17[arg17_xmax]
XmaxDist17_5 = XmaxDist17_5[arg17_5_xmax]
Zenith16_5_xmax = Zenith16_5_xmax[arg16_5_xmax]
Zenith17_xmax = Zenith17_xmax[arg17_xmax]
Zenith17_5_xmax = Zenith17_5_xmax[arg17_5_xmax]


plt.scatter(Zenith16_5_xmax, XmaxDist16_5)
plt.scatter(Zenith17_xmax, XmaxDist17)
plt.scatter(Zenith17_5_xmax, XmaxDist17_5)
plt.show()

plt.scatter(Zenith16_5, Epart16_5)
plt.scatter(Zenith17, Epart17)
plt.scatter(Zenith17_5, Epart17_5)
plt.show()

plt.scatter(XmaxDist16_5, Epart16_5)
plt.scatter(XmaxDist17, Epart17)
plt.scatter(XmaxDist17_5, Epart17_5)
plt.show()