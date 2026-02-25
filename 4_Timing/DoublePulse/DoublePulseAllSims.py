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
from MainModules.PlotConfig import MatplotlibConfig
#sys.path.append("/Users/chiche/Desktop/DeepCrSearch/Analysis/")
##from Modules.SimParam.GetLayoutScaling import GetDepthScaling, GetEnergyScaling
from scipy.interpolate import interp1d
import scipy
##from Modules.Fluence.FunctionsGetFluence import  Norm, LoadTraces, GetPeakTraces, Traces_cgs_to_si, GetDepths, CorrectScaling, CombineTraces, CorrectLength, GetIntTraces, GetIntTracesSum, GetRadioExtent
#from FunctionsPlotFluence import EfieldMap, PlotLDF, PlotTraces, plot_polarisation, PlotMaxTraces, PlotAllTraces, PlotLayer, PlotGivenTrace, PlotAllChannels, PlotSurfaceEz
from ModuleDoubleBumps import GetDoubleBumps, GetNtriggered, GetDoublePulsesMap, GetPulseFlagsData
from ModulePlotDumbleBumps import PlotPeakEfield, PlotDumbleBumpsMaps, PlotDoubleBumpVsZen, PlotTimeDelay, PlotNAirtrigger, PlotNIcetrigger, PlotNtriggAll, PlotNdoubleTot, PlotDoubleRateTot, PlotDoubleRateTotperChannel, PlotDumbleBumpsMapsHighRes
from scipy.interpolate import griddata
from datetime import datetime
from scipy.optimize import curve_fit
from collections import defaultdict
##from Modules.Fluence.FunctionsRadiationEnergy import GetFluence, GetRadiationEnergy
#endregion

#region Path definition
SimDir = "FullDenseDeepCr" #"DeepCrLibV1"  #"InterpSim"
WorkPath = os.getcwd()
simpath = "/Users/chiche/Desktop/DeepCrAnalysis/Simulations/" + SimDir

BatchID = "DoublePulse"
OutputPath = MatplotlibConfig(WorkPath, SimDir, BatchID)
#endregion
Save = False
simpathdir = "/Users/chiche/Desktop/DeepCrAnalysis/Simulations/" + SimDir + "/"
SimpathAll = glob.glob(simpathdir +"*")
Path = SimpathAll


pulse_flags_all = dict()

SignalProp = defaultdict(lambda: defaultdict(dict))
EnergyAll, ZenithAll = [], []
PosDoubleBumpsAll = []
PosDoubleBumpsAll_zen0 = []
PosDoubleBumpsAll_zen50 = []
NtriggerAll_x, NtriggerAll_y, NtriggerAll_z, NtriggerAll = [], [], [], []
k= 0

sigma = 40
threshold1 =5*sigma
threshold2 = 3*sigma

for simpath in SimpathAll:
    print(simpath)
    Shower = CreateShowerfromHDF5(simpath)
    # =============================================================================
    #                              Load Traces
    # =============================================================================

    energy, zenith, Nant = Shower.energy, Shower.zenith, Shower.nant
    Traces_C, Traces_G, Pos = Shower.traces_c, Shower.traces_g, Shower.pos
    Nlay, Nplane, Depths = Shower.GetDepths()

    SelDepth = min(Depths)  #3116 m

    # Initialization
    SignalProp[energy][zenith] = {"Eair": [], "Eice": [], "Pos": []}
    # We skip simulations with issues

    # We focus the study on showers at 10^17.5 eV
    if(energy!=0.1):
        continue
    EnergyAll.append(energy)
    ZenithAll.append(zenith)


    # =============================================================================
    #                                Filter
    # =============================================================================

    Filter = True
    if(Filter):
        fs, lowcut, highcut = 5e9, 50e6, 1e9
        Traces_C =Shower.filter_all_traces(Traces_C, fs, lowcut, highcut)
        Traces_G =Shower.filter_all_traces(Traces_G, fs, lowcut, highcut)

    # =============================================================================
    #                                 Get integral
    # =============================================================================

    # Getting the traces peak amplitude
    # DepthCut
    sel = (Pos[:,2] == SelDepth)#min(Depths))

    Eair_peak = Shower.GetPeakTraces(Traces_C, sel)
    Eice_peak = Shower.GetPeakTraces(Traces_G, sel)
    
    # Extracting the numbers of trigger for each channel
    Ntrigger_x, Ntrigger_y, Ntrigger_z, Ntrigger_tot = \
        GetNtriggered(Eair_peak, Eice_peak, thresold=threshold1)
    NtriggerAll_x.append(Ntrigger_x)
    NtriggerAll_y.append(Ntrigger_y)
    NtriggerAll_z.append(Ntrigger_z)
    NtriggerAll.append(Ntrigger_tot)

    print("Ntriggered tot:", Ntrigger_tot)

    SignalProp[energy][zenith]["Eair"].append(Eair_peak)
    SignalProp[energy][zenith]["Eice"].append(Eice_peak)
    
    # =============================================================================
    #                          Double pulses
    # =============================================================================

    # Finding the double pulse events
    pulse_flags_all[k]= \
    GetDoubleBumps(Shower, Eair_peak, Eice_peak, thresold1=threshold1, thresold2=threshold2, Plot = False)
    DoublePulseFlags = pulse_flags_all[k]["isDoublePulse"]["tot"]

    # Double Bump maps
    PosDoubleBumps = PlotDumbleBumpsMaps(Pos, np.array(DoublePulseFlags), energy, zenith, sel)
    
    PlotDumbleBumpsMapsHighRes(Pos, np.array(DoublePulseFlags), energy, zenith, OutputPath, sel)
    if(zenith >= 0):
        PosDoubleBumpsAll.append(PosDoubleBumps)
    if(zenith ==0):
        PosDoubleBumpsAll_zen0.append(PosDoubleBumps)
    if(zenith ==50):
        PosDoubleBumpsAll_zen50.append(PosDoubleBumps)
    #PosDoubleBumpsAll.append(PosDoubleBumps)
    k = k + 1
    
Nsingleair_x, Nsingleair_y, Nsingleair_z, Nsingleice_x, Nsingleice_y, Nsingleice_z,  Ndouble_x, Ndouble_y, Ndouble_z, Ndouble_tot = \
GetPulseFlagsData(EnergyAll, ZenithAll, Pos, pulse_flags_all, SelDepth)


    
GetDoublePulsesMap(PosDoubleBumpsAll, OutputPath)
GetDoublePulsesMap(PosDoubleBumpsAll_zen0, OutputPath, 0)
GetDoublePulsesMap(PosDoubleBumpsAll_zen50, OutputPath, 50)

NtriggerAll = np.array(NtriggerAll)
Ntrigger_All_x = np.array(NtriggerAll_x)
Ntrigger_All_y = np.array(NtriggerAll_y)
Ntrigger_All_z = np.array(NtriggerAll_z)

selE = 0.1
PlotNAirtrigger(ZenithAll, Nsingleair_x, Nsingleair_y, Nsingleair_z, threshold1, threshold2, selE)
PlotNIcetrigger(ZenithAll, Nsingleice_x, Nsingleice_y, Nsingleice_z, threshold1, threshold2, selE)
PlotNtriggAll(ZenithAll, NtriggerAll)
PlotNdoubleTot(ZenithAll, Ndouble_tot)
PlotDoubleRateTot(ZenithAll, Ndouble_tot, NtriggerAll)
PlotDoubleRateTotperChannel(ZenithAll, Ndouble_x, Ndouble_y, Ndouble_z, Ntrigger_All_x, Ntrigger_All_y, Ntrigger_All_z, OutputPath, sigma)


def PlotDoubleRateTotperChannel(ZenithAll, Ndouble_x, Ndouble_y, Ndouble_z, Ntrigger_All_x, Ntrigger_All_y, Ntrigger_All_z, OutputPath, sigma):

    colors = ["#0072B2", "#E69F00", "#009E73"]  # Blue, Orange, Green (colorblind-safe)
    linestyles = ["-", "--", "-."]

    arg = np.argsort(ZenithAll)
    Ndouble_x =Ndouble_x[~np.isnan(Ntrigger_All_x)]
    Ntrigger_All_x = Ntrigger_All_x[~np.isnan(Ntrigger_All_x)]
    plt.plot(np.array(ZenithAll)[arg], Ndouble_x[arg]/Ntrigger_All_x[arg], label ="x", color=colors[0], linestyle=linestyles[0], linewidth=2, marker='o', markersize=5)
    plt.plot(np.array(ZenithAll)[arg], Ndouble_y[arg]/Ntrigger_All_y[arg], label ="y", color=colors[1], linestyle=linestyles[1], linewidth=2, marker='o', markersize=5)
    plt.plot(np.array(ZenithAll)[arg], Ndouble_z[arg]/Ntrigger_All_z[arg], label="z", color=colors[2], linestyle=linestyles[2], linewidth=2, marker='o', markersize=5)
    plt.xlabel("Zenith [Deg.]")
    plt.ylabel(r"$N_{\mathrm{double}}/N_{\mathrm{trigger}}$")
    plt.title(r"$E = 10^{17}\,\mathrm{eV},\ \sigma = %d\,\mu\mathrm{V/m}$" % sigma,
          fontsize=14)    
    plt.legend()
    plt.grid(True, which='both', linestyle=':', linewidth=0.5)
    plt.savefig(OutputPath + "DoubleRateAllchannels.pdf", bbox_inches="tight")
    plt.show()

PlotDoubleRateTotperChannel(ZenithAll, Ndouble_x, Ndouble_y, Ndouble_z, Ntrigger_All_x, Ntrigger_All_y, Ntrigger_All_z, OutputPath, sigma)

sys.exit()
bin_edges = np.linspace(0, 100, 41) 
labels=('x', 'y', 'z')
plt.hist(SignalProp[0.316][0]["Eair"][0][0], bin_edges, alpha=0.6, edgecolor='black', label=labels[0])
plt.hist(SignalProp[0.316][0]["Eair"][0][1], bin_edges, alpha=0.6, edgecolor='black', label=labels[1])
plt.hist(SignalProp[0.316][0]["Eair"][0][2], bin_edges, alpha=0.6, edgecolor='black', label=labels[2])
plt.xlabel('$E_{int}\, [50-1000 MHz]\, (\mu Vs /m)$')
plt.ylabel('Number of antennas')
#plt.xlim(0,2000)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.title(r"In-air, $\theta =0^{\circ}$, $E=10^{17.5} eV$")
#plt.savefig("/Users/chiche/Desktop/InAirFilteredPulseDistrib.pdf", bbox_inches="tight")
plt.show()

Eice_x = SignalProp[0.316][0]["Eice"][0][0]
Eice_y = SignalProp[0.316][0]["Eice"][0][1]
Eice_z = SignalProp[0.316][0]["Eice"][0][2]

plt.hist(Eice_x, bins=50, range=(0, int(1e5)), alpha=0.6, edgecolor='black', label=labels[0])
plt.hist(Eice_y, bins=50, alpha=0.6, range=(0, int(1e5)), edgecolor='black', label=labels[1])
plt.hist(Eice_z, bins=50, alpha=0.6, range=(0, int(1e5)), edgecolor='black', label=labels[2])
plt.xlabel('$E_{int}\, [50-1000 MHz]\, (\mu Vs /m)$')
plt.ylabel('Number of antennas')
plt.xlim(0,4000)
#plt.xscale("log")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.title(r"In-ice, $\theta =0^{\circ}$, $E=10^{17.5} eV$")
#plt.savefig("/Users/chiche/Desktop/InIceFilteredPulseDistrib.pdf", bbox_inches="tight")
plt.show()







# Ntrigger for each emission vs zenith and depth


Nsingleairlayers_x, Nsingleairlayers_y, Nsingleairlayers_z = dict(), dict(), dict()  
Nsingleicelayers_x, Nsingleicelayers_y, Nsingleicelayers_z = dict(), dict(), dict()  
Ndoublelayers_x, Ndoublelayers_y, Ndoublelayers_z = dict(), dict(), dict()  


for j in range(len(Depths)):
    sel = (Pos[:,2] == Depths[j])
    Ndoublelayer_x, Ndoublelayer_y, Ndoublelayer_z = np.zeros(len(EnergyAll)), np.zeros(len(EnergyAll)), np.zeros(len(EnergyAll))
    Nsingleairlayer_x, Nsingleairlayer_y, Nsingleairlayer_z = np.zeros(len(EnergyAll)), np.zeros(len(EnergyAll)), np.zeros(len(EnergyAll))
    Nsingleicelayer_x, Nsingleicelayer_y, Nsingleicelayer_z = np.zeros(len(EnergyAll)), np.zeros(len(EnergyAll)), np.zeros(len(EnergyAll))

    for i in range(len(EnergyAll)):
        isAirSinglePulse, isIceSinglePulse, isDoublePulse, Deltat = \
            (pulse_flags_all[i][key] for key in ["isAirSinglePulse", "isIceSinglePulse", "isDoublePulse", "Deltat"])
            
        Nsingleairlayer_x[i] = np.sum(np.array(isAirSinglePulse["x"])[sel])
        Nsingleairlayer_y[i] = np.sum(np.array(isAirSinglePulse["y"])[sel])
        Nsingleairlayer_z[i] = np.sum(np.array(isAirSinglePulse["z"])[sel])

        Nsingleicelayer_x[i] = np.sum(np.array(isIceSinglePulse["x"])[sel])
        Nsingleicelayer_y[i] = np.sum(np.array(isIceSinglePulse["y"])[sel])
        Nsingleicelayer_z[i] = np.sum(np.array(isIceSinglePulse["z"])[sel])

        Ndoublelayer_x[i] = np.sum(np.array(isDoublePulse["x"])[sel])
        Ndoublelayer_y[i] = np.sum(np.array(isDoublePulse["y"])[sel])
        Ndoublelayer_z[i] = np.sum(np.array(isDoublePulse["z"])[sel])
    
    Nsingleairlayers_x[j] = Nsingleairlayer_x
    Nsingleairlayers_y[j] = Nsingleairlayer_y
    Nsingleairlayers_z[j] = Nsingleairlayer_z

    Nsingleicelayers_x[j] = Nsingleicelayer_x
    Nsingleicelayers_y[j] = Nsingleicelayer_y
    Nsingleicelayers_z[j] = Nsingleicelayer_z

    Ndoublelayers_x[j]  = Ndoublelayer_x
    Ndoublelayers_y[j]  = Ndoublelayer_y
    Ndoublelayers_z[j]  = Ndoublelayer_z

for k in range(len(Depths)):

    plt.scatter(ZenithAll, Nsingleairlayers_x[k], label ="x")
    plt.scatter(ZenithAll, Nsingleairlayers_y[k], label ="y")
    plt.scatter(ZenithAll, Nsingleairlayers_z[k], label="z")
    plt.xlabel("zenith [Deg.]")
    plt.ylabel("$N_{trigger}^{air}$")
    plt.title("$E=10^{17.5} eV$, $Depth = %.d$ m" %Depths[k])
    plt.legend()
    #plt.savefig("/Users/chiche/Desktop/Ntrigair_E0.316_Depth%.d_zen.pdf" %Depths[k])
    plt.show()

    plt.scatter(ZenithAll, Nsingleicelayers_x[k], label ="x")
    plt.scatter(ZenithAll, Nsingleicelayers_y[k], label ="y")
    plt.scatter(ZenithAll, Nsingleicelayers_z[k], label="z")
    plt.xlabel("zenith [Deg.]")
    plt.ylabel("$N_{trigger}^{ice}$")
    plt.title("$E=10^{17.5} eV$, $Depth = %.d$ m" %Depths[k])
    plt.legend()
    #plt.savefig("/Users/chiche/Desktop/Ntrigice_E0.316_Depth%.d_zen.pdf" %Depths[k])
    plt.show()

    Ntriggeredlayerx = Nsingleairlayers_x[k] + Nsingleicelayers_x[k] - Ndoublelayers_x[k]
    Ntriggeredlayery = Nsingleairlayers_y[k] + Nsingleicelayers_y[k] - Ndoublelayers_y[k]
    Ntriggeredlayerz = Nsingleairlayers_z[k] + Nsingleicelayers_z[k] - Ndoublelayers_z[k]

    DoubleRatelayer_x =  Ndoublelayers_x[k]/Ntriggeredlayerx
    DoubleRatelayer_y =  Ndoublelayers_y[k]/Ntriggeredlayery
    DoubleRatelayer_z =  Ndoublelayers_z[k]/Ntriggeredlayerz

    plt.scatter(ZenithAll, DoubleRatelayer_x, label ="x")
    plt.scatter(ZenithAll, DoubleRatelayer_y, label ="y")
    plt.scatter(ZenithAll, DoubleRatelayer_z, label="z")
    plt.xlabel("zenith [Deg.]")
    plt.ylabel("$N_{double}/N_{triggered}$")
    plt.title("$E=10^{17.5} eV$, $th1 = 600 \, \mu Vs/m$, $th2 = 400 \, \mu Vs/m$")
    plt.legend()
    #plt.savefig("/Users/chiche/Desktop/DoubleRate_E0.316_Depth%.d_vs_zen.pdf" %Depths[k], bbox_inches = "tight")
    plt.show()

