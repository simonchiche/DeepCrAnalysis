#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 02:21:35 2024

@author: chiche
"""

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
#sys.path.append("/Users/chiche/Desktop/DeepCrSearch/Analysis/")
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
SimDir = "DeepCrLib"  #"InterpSim"
SimName = "Rectangle_Proton_0.316_0_0_1"
WorkPath = os.getcwd()
simpath = "/Users/chiche/Desktop/DeepCrAnalysis/Simulations/DeepCrLibV1/"\
      + SimName + "_0.hdf5"
BatchID = "Linear"
OutputPath = MatplotlibConfig(WorkPath, SimDir, BatchID)
#endregion
Save = False

SimFolder = "/Users/chiche/Desktop/DeepCrAnalysis/Simulations/DeepCrLibV1"
PathAll = glob.glob(SimFolder + "/*")

Lextentx_air, Lmaxx_air= np.zeros(len(PathAll)), np.zeros(len(PathAll))
Lextenty_air, Lmaxy_air= np.zeros(len(PathAll)), np.zeros(len(PathAll))
Lextentx100_air, Lmaxx100_air= np.zeros(len(PathAll)), np.zeros(len(PathAll))
Lextenty100_air, Lmaxy100_air= np.zeros(len(PathAll)), np.zeros(len(PathAll))

Lextentx_ice, Lmaxx_ice= np.zeros(len(PathAll)), np.zeros(len(PathAll))
Lextenty_ice, Lmaxy_ice= np.zeros(len(PathAll)), np.zeros(len(PathAll))
Lextentx100_ice, Lmaxx100_ice= np.zeros(len(PathAll)), np.zeros(len(PathAll))
Lextenty100_ice, Lmaxy100_ice= np.zeros(len(PathAll)), np.zeros(len(PathAll))


ZenithAll = np.zeros(len(PathAll))

k =0
for simpath in PathAll:
    Shower = CreateShowerfromHDF5(simpath)
    # =============================================================================
    #                              Load Traces
    # =============================================================================

    energy, theta, Nant = Shower.energy, Shower.zenith, Shower.nant
    Traces_C, Traces_G, Pos = Shower.traces_c, Shower.traces_g, Shower.pos
    Nlay, Nplane, Depths = Shower.GetDepths()
    #Traces_tot = Shower.CombineTraces()
    ZenithAll[k] = Shower.zenith

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
    #            Parametrization of the spacing (#THIS PART NEEDS UPDATES!)
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

    # Surface antennas results x-channel
    posx_MaxAirLDF, Ex_MaxAirLDF = GetMaxLDF(Pos, EtotC_int, Depths[0], "x") 
    posx_MaxIceLDF, Ex_MaxIceLDF = GetMaxLDF(Pos, EtotG_int, Depths[0], "x") 

    #Lextent: extent for which we have 99% of the total energy
    # Lmax: extent at which the signal reaches its maximum
    Lextentx_air[k], Lmaxx_air[k] = GetCaracExtent(posx_MaxAirLDF[:,0], Ex_MaxAirLDF)
    Lextentx_ice[k], Lmaxx_ice[k] = GetCaracExtent(posx_MaxIceLDF[:,0], Ex_MaxIceLDF)

    # Surface antennas results y-channel
    posy_MaxAirLDF, Ey_MaxAirLDF = GetMaxLDF(Pos, EtotC_int, Depths[0], "y") 
    posy_MaxIceLDF, Ey_MaxIceLDF = GetMaxLDF(Pos, EtotG_int, Depths[0], "y") 
    Lextenty_air[k], Lmaxy_air[k] = GetCaracExtent(posy_MaxAirLDF[:,1], Ey_MaxAirLDF)
    Lextenty_ice[k], Lmaxy_ice[k] = GetCaracExtent(posx_MaxIceLDF[:,1], Ey_MaxIceLDF)

    # Deep antennas results x-channel
    posx100_MaxAirLDF, Ex100_MaxAirLDF = GetMaxLDF(Pos, EtotC_int, Depths[4], "x") 
    posx100_MaxIceLDF, Ex100_MaxIceLDF = GetMaxLDF(Pos, EtotG_int, Depths[4], "x") 
    Lextentx100_air[k], Lmaxx100_air[k] = GetCaracExtent(posx100_MaxAirLDF[:,0], Ex100_MaxAirLDF)
    Lextentx100_ice[k], Lmaxx100_ice[k] = GetCaracExtent(posx100_MaxIceLDF[:,0], Ex100_MaxIceLDF)

    # Deep antennas results y-channel
    posy100_MaxAirLDF, Ey100_MaxAirLDF = GetMaxLDF(Pos, EtotC_int, Depths[4], "y") 
    posy100_MaxIceLDF, Ey100_MaxIceLDF = GetMaxLDF(Pos, EtotG_int, Depths[4], "y") 
    Lextenty100_air[k], Lmaxy100_air[k] = GetCaracExtent(posy100_MaxAirLDF[:,1], Ey100_MaxAirLDF)
    Lextenty100_ice[k], Lmaxy100_ice[k] = GetCaracExtent(posy100_MaxIceLDF[:,1], Ey100_MaxIceLDF)


    # =============================================================================
    #                              PLOTS
    # =============================================================================
    #PlotAirIceExtent(Depths, airextent, icextent, simextent, energy, theta)
    #PlotRadioSimExtent(Depths, radioextent, simextent)
    #PlotFillingFactor(Depths, radioextent, simextent)

    #PlotMaxLDF(posx_MaxAirLDF, posx_MaxIceLDF, Ex_MaxAirLDF, Ex_MaxIceLDF, "x", Depths[0], Shower, OutputPath)
    #PlotMaxLDF(posy_MaxAirLDF, posy_MaxIceLDF, Ey_MaxAirLDF, Ey_MaxIceLDF, "y",Depths[0], Shower, OutputPath)
    #PlotMaxLDF(posx100_MaxAirLDF, posx100_MaxIceLDF, Ex100_MaxAirLDF, Ex100_MaxIceLDF, "x", Depths[4], Shower, OutputPath)
    #PlotMaxLDF(posy100_MaxAirLDF, posy100_MaxIceLDF, Ey100_MaxAirLDF, Ey100_MaxIceLDF, "y",Depths[4], Shower, OutputPath)

    k = k +1

# 99% extent along the x and y directions for all the energy bins for surface antennas in-air
plt.scatter(ZenithAll, Lextentx_air, label = "x")
plt.scatter(ZenithAll, Lextenty_air, label = "y")
plt.xlabel("Zenith [Deg.]")
plt.ylabel("L99 [m]")
plt.legend()
plt.title("In-air emission surface antennas")
#plt.savefig(path + "L95air.pdf")
plt.show()

# 99% extent along the x and y directions for all the energy bins for 100m-deep antennas in-air
plt.scatter(ZenithAll, Lextentx100_air, label = "x")
plt.scatter(ZenithAll, Lextenty100_air, label = "y")
plt.xlabel("Zenith [Deg.]")
plt.ylabel("L99 [m]")
plt.legend()
plt.title("In-air emission 100m deep")
#plt.savefig(path + "L95air100.pdf")
plt.show()

# 99% extent along the x and y directions for all the energy bins for 100m-deep antennas in-ice
plt.scatter(ZenithAll, Lextentx100_ice, label = "x")
plt.scatter(ZenithAll, Lextenty100_ice, label = "y")
plt.xlabel("Zenith [Deg.]")
plt.ylabel("L99 [m]")
plt.legend()
plt.title("In-ice emission 100m deep")
#plt.savefig(path + "L95100ice.pdf")
plt.show()

# Extent at the Cerenkov cone along the x and y directions 
# #for all the energy bins for surface antennas in-air
plt.scatter(ZenithAll, Lmaxx_air, label = "x")
plt.scatter(ZenithAll, Lmaxy_air, label = "y")
plt.xlabel("Zenith [Deg.]")
plt.ylabel("Lmax [m]")
plt.title("In-air emission surface antennas")
plt.legend()
#plt.savefig(path + "Lmaxair.pdf")
plt.show()

# Extent at the Cerenkov cone along the x and y directions 
# #for all the energy bins for 100m-deep antennas in-air
plt.scatter(ZenithAll, Lmaxx100_air, label = "x")
plt.scatter(ZenithAll, Lmaxy100_air, label = "y")
plt.xlabel("Zenith [Deg.]")
plt.ylabel("Lmax [m]")
plt.title("In-air emission 100m deep")
plt.legend()
#plt.savefig(path + "Lmax100air.pdf")
plt.show()

# Extent at the Cerenkov cone along the x and y directions 
# #for all the energy bins for 100m-deep antennas in-ice
plt.scatter(ZenithAll, Lmaxx100_ice, label = "x")
plt.scatter(ZenithAll, Lmaxy100_ice, label = "y")
plt.xlabel("Zenith [Deg.]")
plt.ylabel("Lmax [m]")
plt.title("In-ice emission 100m deep")
plt.legend()
#plt.savefig(path + "Lmax100ice.pdf")
plt.show()



# =============================================================================
#                         Extent tests
# =============================================================================

#Lextentx60_air = (Lextentx100_air + Lextentx_air)/2.0
#LextentData = np.concatenate([Lextentx_air, Lextentx60_air, Lextentx100_air])
#ZenithData = np.concatenate([ZenithAll, ZenithAll, ZenithAll])
#Depth1 = np.ones(len(Lextentx_air))*Depths[0]
#Depth2 = np.ones(len(Lextentx_air))*Depths[2]
#Depth3 = np.ones(len(Lextentx_air))*Depths[4]
#DepthData = np.concatenate([Depth1, Depth2,Depth3 ])
#ExtentData = np.array([LextentData, ZenithData, DepthData]).T

#plt.scatter(ExtentData[:,1], ExtentData[:,0])
#plt.show()
#np.savetxt("./Data/ExtentData.txt", ExtentData)








