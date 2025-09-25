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
from Modules.ModuleSimExtent import GetRadioExtent, GetMaxLDF, GetMaxLDFx, GetCaracExtent
from Modules.ModulePlotLDFs import PlotAllAirLdfs, PlotAllIceLdfs, PlotAllAirLdfsGeneric, PlotAllIceLdfsGeneric
##from Modules.Fluence.FunctionsRadiationEnergy import GetFluence, GetRadiationEnergy
#endregion

#region Path definition
SimDir = "FullDenseDeepCr" #"DeepCrLibV1"  #"InterpSim"
SimName = "Polar_Proton_0.316_0_0_1.hdf5" #"Rectangle_Proton_0.316_0_0_1"
WorkPath = os.getcwd()
simpath = "/Users/chiche/Desktop/DeepCrAnalysis/Simulations/" + SimDir 
SimpathAll = glob.glob(simpath + "/*")
BatchID = "LDFAllSims"
OutputPath = MatplotlibConfig(WorkPath, SimDir, BatchID)
#endregion
Save = True

posx_MaxAirLDFAll, posx_MaxIceLDFAll, Ex_MaxAirLDFAll, Ex_MaxIceLDFAll, EnergyAll, ZenithAll = \
     ([] for _ in range(6))

for simpath in SimpathAll:
    print(simpath.split("/")[-1])
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
    posx_MaxAirLDF, Ex_MaxAirLDF = GetMaxLDFx(Pos, EtotC_int, Depths[0], "x") 
    posx_MaxIceLDF, Ex_MaxIceLDF = GetMaxLDFx(Pos, EtotG_int, Depths[0], "x") 


    # 60m-deep antennas results
    posx60_MaxAirLDF, Ex60_MaxAirLDF = GetMaxLDFx(Pos, EtotC_int, Depths[1], "x") 
    posx60_MaxIceLDF, Ex60_MaxIceLDF = GetMaxLDFx(Pos, EtotG_int, Depths[1], "x") 


    # 100m-deep antennas results
    posx100_MaxAirLDF, Ex100_MaxAirLDF = GetMaxLDFx(Pos, EtotC_int, Depths[2], "x") 
    posx100_MaxIceLDF, Ex100_MaxIceLDF = GetMaxLDFx(Pos, EtotG_int, Depths[2], "x") 

    arg= np.argsort(posx100_MaxIceLDF[:, 0])
    posx100_MaxIceLDF = posx100_MaxIceLDF[arg]
    Ex100_MaxIceLDF = Ex100_MaxIceLDF[arg]
    plt.plot(posx100_MaxIceLDF[:, 0], Ex100_MaxIceLDF, "-o")
    plt.xlabel("Position [m]")
    plt.ylabel("$E_{tot}^{int}$ [$\mu$V/m]")
    plt.title("Ice, E=%.2f EeV, Depth =$%d\,m$" %(energy, Depths[2]))
    plt.grid()
    plt.savefig(OutputPath + "Ice_LDF_%.3f_EeV_Depth%d_zen%.d.pdf" %(energy, Depths[2], theta), bbox_inches='tight')           
    plt.show()
   


    for lst, val in zip([posx_MaxAirLDFAll, posx_MaxIceLDFAll, Ex_MaxAirLDFAll, Ex_MaxIceLDFAll, EnergyAll, ZenithAll],
                        [posx100_MaxAirLDF, posx100_MaxIceLDF, Ex100_MaxAirLDF, Ex100_MaxIceLDF, energy, theta]):
        lst.append(val)


posx_MaxAirLDFAll, posx_MaxIceLDFAll, Ex_MaxAirLDFAll, Ex_MaxIceLDFAll, EnergyAll, ZenithAll = \
map(np.array, [posx_MaxAirLDFAll, posx_MaxIceLDFAll, Ex_MaxAirLDFAll, Ex_MaxIceLDFAll, EnergyAll, ZenithAll])

selE = 0.316
selDepth = 100

PlotAllAirLdfsGeneric(posx_MaxAirLDFAll, Ex_MaxAirLDFAll, ZenithAll, EnergyAll, selE, selDepth, "Air", OutputPath, Save=True)
PlotAllIceLdfsGeneric(posx_MaxIceLDFAll, Ex_MaxIceLDFAll, ZenithAll, EnergyAll, selE, selDepth, "Ice", OutputPath, Save=True)


