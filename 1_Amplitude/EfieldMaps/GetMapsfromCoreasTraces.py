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
from Modules.FunctionsPlotFluence import EfieldMap, PlotLDF, PlotTraces, plot_polarisation, PlotMaxTraces, PlotAllTraces, PlotLayer, PlotGivenTrace, PlotAllChannels, PlotSurfaceEz
##from CleanCoreasTraces import CleanCoreasTraces
##from Modules.SimParam.PlotRadioSimExtent import PlotFillingFactor, PlotRadioSimExtent, PlotAirIceExtent
from scipy.interpolate import griddata
from datetime import datetime
from scipy.optimize import curve_fit
from Modules.ModuleGetCoreasMaps import GetCoreasTracesfromHDF5, PlotCoreasMaps
from MainModules.FormatFaerieOutput import Traces_cgs_to_si
from matplotlib.colors import PowerNorm
from scipy.interpolate import Rbf
from scipy.interpolate import griddata
##from Modules.Fluence.FunctionsRadiationEnergy import GetFluence, GetRadiationEnergy
#endregion

#region Path definition
SimDir = "DenseDeepCr"  #"InterpSim"
WorkPath = os.getcwd()
BatchID = "Coreas"
#simpath = "/Users/chiche/Desktop/DeepCrAnalysis/hdf5FAERIE/DenseDeepCr/Polar_Proton_0.316_0_0_1.hdf5" 
simpath = "/Users/chiche/Desktop/DeepCrAnalysis/Simulations/DeepCrLibV1/Rectangle_Proton_0.316_43_0_1_0.hdf5" 
OutputPath = MatplotlibConfig(WorkPath, SimDir, BatchID)
#endregion
Save = False

Shower = CreateShowerfromHDF5(simpath)
Shower.traces_c = GetCoreasTracesfromHDF5(simpath)
Shower.traces_c = Traces_cgs_to_si(Shower.traces_c)

Filter = False
if(Filter):
    fs, lowcut, highcut = 5e9, 50e6, 1e9
    Shower.traces_c =Shower.filter_all_traces(Shower.traces_c, fs, lowcut, highcut)

ExC, EyC, EzC, EtotC, peakTime = Shower.GetIntTraces(Shower.traces_c)

PlotCoreasMaps(Shower, EtotC)


### Integral
def interpolate_rbf(x, y, z, grid_resolution=100, bounds=None, function='cubic'):
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    if bounds is None:
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
    else:
        xmin, xmax, ymin, ymax = bounds

    if isinstance(grid_resolution, int):
        nx = ny = grid_resolution
    else:
        nx, ny = grid_resolution

    grid_x, grid_y = np.meshgrid(
        np.linspace(xmin, xmax, nx),
        np.linspace(ymin, ymax, ny)
    )

    rbf = Rbf(x, y, z, function=function)  # function='linear', 'multiquadric', 'gaussian', etc.
    grid_z = rbf(grid_x, grid_y)

    return grid_x, grid_y, grid_z

from scipy.signal import hilbert
from scipy.integrate import trapz, simps

def GetFluence(Traces):
    eps0 = 8.85e-12 # F.m^{-1}
    c = 3e8 # m.s^{-1}

    Nant = len(Traces)
    
    ftot = np.zeros(Nant)
    fx, fy, fz = np.zeros(Nant), np.zeros(Nant), np.zeros(Nant)
    binT = round((Traces[0][1,0] -Traces[0][0,0])*1e10)/1e10
    print(binT)

    for i in range(Nant):
        
        ftot_t = Traces[i][:,1]**2 + Traces[i][:,2]**2 + Traces[i][:,3]**2
        extent = 10000
        peak_id = np.argmax(ftot_t)
        minid = peak_id -extent
        maxid = peak_id + extent
        if(minid<0): minid = 0
        if(maxid>len( Traces[i][:,0])): maxid =len( Traces[i][:,0])
        
        time = np.arange(0, len(Traces[i][minid:maxid,0]))*binT
        
        fx[i] = eps0*c*simps(abs(hilbert(Traces[i][minid:maxid,1]**2)), time)/1e12
        fy[i] = eps0*c*simps(abs(hilbert(Traces[i][minid:maxid,2]**2)), time)/1e12
        fz[i] = eps0*c*simps(abs(hilbert(Traces[i][minid:maxid,3]**2)), time)/1e12
        ftot[i] = eps0*c*(fx[i] + fy[i] + fz[i])
    print(fy,ftot)
    return fx, fy, fz, ftot

Pos = Shower.pos
Nlay, Nplane, Depths = Shower.GetDepths()
fx, fy, fz, ftot = GetFluence(Shower.traces_c)

AllInt = np.zeros(len(Depths))
from scipy.integrate import trapz
for i in range(len(Depths)):
    sel = (Pos[:,2] == Depths[i])

    grid_x, grid_y, grid_z = \
    interpolate_rbf(Pos[:,0][sel], Pos[:,1][sel], ftot[sel])
    '''
    plt.figure(figsize=(6, 5))
    plt.contourf(grid_x, grid_y, np.log10(grid_z), levels=100, cmap='jet')
    plt.scatter(Pos[:,0][sel], Pos[:,1][sel], s =0.1)
    #plt.scatter(Pos[:729,0], Pos[:729,1], c=np.log10(EtotC_int[:729] +1), edgecolor='white', s=100)
    plt.colorbar(label="$\log_{10}()$ [$\mu Vs/m$]")
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.show()
    '''

    # First, compute the integral along one axis (e.g., x), then along the other (e.g., y)
    integral_x = trapz(grid_z, x=grid_x[0], axis=1)  # integrate over x (axis=1)
    total_integral = trapz(integral_x, x=grid_y[:,0])  # integrate over y (axis=0)
    AllInt[i] = total_integral


plt.scatter(Depths, AllInt)
#plt.ylim(int(2e7), int(3.3e7))
plt.xlabel("Depth [m]")
plt.ylabel("$E_{rad}$")
plt.grid()
#plt.savefig("/Users/chiche/Desktop/Rectangle_Erad_vs_depth_E0.32_th43.pdf", bbox_inches = "tight")
plt.show()