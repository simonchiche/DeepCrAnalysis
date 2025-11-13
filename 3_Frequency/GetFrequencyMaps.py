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
from Modules.ModuleFrequencySpectrum import compute_spectrum, PlotAllSpectra, PlotFrequencyHeatmap, PlotAllSignals, GetPeakTraces
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

SurfacePos = Pos[Pos[:,2]==Depths[2]]
selsurface = Pos[:,2]==Depths[2]
Trace_C_Surface = {k: v for (k, v), m in zip(Traces_C.items(), selsurface) if m}
Trace_C_Surface = {i: v for i, v in enumerate(Trace_C_Surface.values())}
Trace_G_Surface = {k: v for (k, v), m in zip(Traces_G.items(), selsurface) if m}
Trace_G_Surface = {i: v for i, v in enumerate(Trace_G_Surface.values())}


PlotAllSpectra(Trace_C_Surface)
PlotAllSpectra(Trace_G_Surface)

radius = np.sqrt(SurfacePos[:,0]**2 + SurfacePos[:,1]**2) 
radius_idx = np.argsort(radius)

### Testing ###
radius_bins = np.linspace(0, 740, 50)
for i in range(len(radius_bins)-1):
    mask_rad = (radius >= radius_bins[i]) & (radius < radius_bins[i+1])
    Trace_G_Surface_bin = {k: v for (k, v), m in zip(Trace_G_Surface.items(), mask_rad) if m}
    Trace_G_Surface_bin = {j: v for j, v in enumerate(Trace_G_Surface_bin.values())}
    print(f"Radius bin: {radius_bins[i]} - {radius_bins[i+1]} m, N_antennas: {len(Trace_G_Surface_bin)}")
    if len(Trace_G_Surface_bin) > 0:
        PlotAllSpectra(Trace_G_Surface_bin)
        PlotAllSignals(Trace_G_Surface_bin)
#################

Trigger = False
Threshold = 0
if(Trigger):    
    Ex, Ey, Ez, Etot_C = GetPeakTraces(Trace_C_Surface)
    selE = Etot_C>Threshold
    Trace_C_Surface_highE = {k: v for (k, v), m in zip(Trace_G_Surface.items(), selE) if m}
    Trace_C_Surface_highE = {i: v for i, v in enumerate(Trace_C_Surface_highE.values())}
    radius_highE = radius[selE]
    radius_idx = np.argsort(radius_highE)


PlotFrequencyHeatmap(Trace_C_Surface, radius, radius_idx, Shower, OutputPath, label="In-air", rmax=200, Save=True)
import sys
sys.exit()
def EulerIntegral(Signal, X):

    dx = X[1] - X[0]
    integral = 0
    for i in range(len(Signal)-1):

        integral =integral + Signal[i]*dx
    
    return integral

from scipy.integrate import simpson

freqPower = np.zeros(len(Trace_C_Surface))

for i in range(len(Trace_C_Surface)):
    time = Trace_C_Surface[i][:,0]
    signal =Trace_G_Surface[i][:,2] 
    f, A_x = compute_spectrum(time, Trace_C_Surface[i][:,1] , window='rect',     detrend='none',    # do not remove mean
        onesided=False     # full FFT; we'll slice positive half like you did
    )
    f, A_y = compute_spectrum(time, Trace_C_Surface[i][:,1], window='rect',     detrend='none',    # do not remove mean
        onesided=False     # full FFT; we'll slice positive half like you did
    )
    f, A_z = compute_spectrum(time, Trace_C_Surface[i][:,3], window='rect',     detrend='none',    # do not remove mean
        onesided=False     # full FFT; we'll slice positive half like you did
    )

    #FreqWeightInt = EulerIntegral(A*f, f)
    #NormInt = EulerIntegral(A, f)
    #freqPower[i] = FreqWeightInt/NormInt

    Px, Py, Pz = A_x**2, A_y**2, A_z**2
    Ptot = Px + Py + Pz

    FreqWeightInt = simpson(y=Ptot*f, x=f)
    NormInt = simpson(y=Ptot, x=f)
    freqPower[i] = FreqWeightInt/NormInt
    



plt.scatter(SurfacePos[:,0], SurfacePos[:,1], c=freqPower/max(freqPower), cmap="jet")
cbar = plt.colorbar()
#plt.ylim(-200,200)
#plt.xlim(-200,200)

plt.show()



SurfacePos = Pos[Pos[:,2]==Depths[2]]
Pos60=  Pos[Pos[:,2]==Depths[1]]

plt.scatter(Pos60[:,0], Pos60[:,1], c='b', label='Surface', s =1)
plt.scatter(SurfacePos[:,0], SurfacePos[:,1], c='b', label='Surface', s=1)
