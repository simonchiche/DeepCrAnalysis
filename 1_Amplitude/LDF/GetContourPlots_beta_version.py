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
SimDir = "DeepCrLib"  #"InterpSim"
SimName = "Rectangle_Proton_0.316_50_0_1"
WorkPath = os.getcwd()
simpath = "/Users/chiche/Desktop/DeepCrAnalysis/Simulations/DeepCrLibV1/"\
      + SimName + "_0.hdf5"
BatchID = "Linear"
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
Traces_tot = Shower.CombineTraces()

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
Extot, Eytot, Eztot, Etot_peak = Shower.GetPeakTraces(Traces_tot)

# =============================================================================
#                                 Get integral
# =============================================================================

ExC_int, EyC_int, EzC_int, EtotC_int, peakTime = Shower.GetIntTraces(Traces_C)
ExG_int, EyG_int, EzG_int, EtotG_int, peakTime = Shower.GetIntTraces(Traces_G)
Ex_tot_int, Ey_tot_int, Ez_tot_int, Etot_int, peakTime = Shower.GetIntTraces(Traces_tot)


# =============================================================================
#                            Cleaning the data
# =============================================================================

#region remove core antennas
#Pos, EtotC_int, Ex_tot_int, Ey_tot_int, Ez_tot_int \
#    = RemoveCoreAntennas(Pos, 15, Ex_tot_int, Ey_tot_int, Ez_tot_int, EtotC_int)
#endregion

# =============================================================================
#                         Get contour plots
# =============================================================================

"""
### Method 1 ###

# Suppose x, y, I are 1D and form a grid
# Turn them into 2D if necessary
x = Pos[:729, 0]
y = Pos[:729, 1]
I = EtotC_int[:729]

r = np.sqrt(x**2 + y**2)
sorted_indices = np.argsort(r)
I_sorted = I[sorted_indices]
r_sorted = r[sorted_indices]

I_cumsum = np.cumsum(I_sorted)
I_total = I_cumsum[-1]

threshold_idx = np.searchsorted(I_cumsum, 0.9 * I_total)
r_threshold = r_sorted[threshold_idx]


fig, ax = plt.subplots()

# Plot filled contours using scattered data
tcf = ax.tricontourf(x, y, np.log(I), levels=100, cmap='jet')

# Overlay the 70% intensity region as a circle
circle = plt.Circle((0, 0), r_threshold, color='cyan', fill=False, linewidth=2, label='90% intensity region')
ax.add_patch(circle)

ax.set_aspect('equal')
ax.set_title('Region containing 90% of total intensity')
#ax.legend()
plt.colorbar(tcf, ax=ax, label='Intensity')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


### Method 2 ###

# Data
x = Pos[:729, 0]
y = Pos[:729, 1]
I = EtotC_int[:729]

# Convert to polar coordinates
r = np.sqrt(x**2 + y**2)
theta = np.arctan2(y, x)  # Range: [-π, π]
theta = (theta + 2*np.pi) % (2*np.pi)  # Convert to [0, 2π]

# Total intensity for global reference
I_total = np.sum(I)
I_target = 0.9 * I_total

# Angular bins: 8 sectors of 45°
n_sectors = 8
angles = np.linspace(0, 2*np.pi, n_sectors + 1)

# Store r limits for each direction
r_limits = []
angle_centers = []

for i in range(n_sectors):
    theta_min = angles[i]
    theta_max = angles[i+1]
    theta_center = (theta_min + theta_max) / 2

    # Select points in the angular sector
    mask = (theta >= theta_min) & (theta < theta_max)
    r_sector = r[mask]
    I_sector = I[mask]

    # Sort by radius
    sorted_indices = np.argsort(r_sector)
    r_sorted = r_sector[sorted_indices]
    I_sorted = I_sector[sorted_indices]

    # Cumulative intensity in this sector
    I_cumsum = np.cumsum(I_sorted)

    if I_cumsum.size == 0:
        r_theta = 0
    else:
        # Proportional target in this sector
        sector_total = np.sum(I_sector)
        target = min(I_target * (sector_total / I_total), np.sum(I_sector))  # Safe cap
        threshold_idx = np.searchsorted(I_cumsum, target)
        r_theta = r_sorted[min(threshold_idx, len(r_sorted) - 1)]

    r_limits.append(r_theta)
    angle_centers.append(theta_center)


# Close the polar contour
angle_centers = np.array(angle_centers)
r_limits = np.array(r_limits)

# Add the first point to close the loop
angle_closed = np.append(angle_centers, angle_centers[0])
r_closed = np.append(r_limits, r_limits[0])

# Convert back to Cartesian coordinates
x_contour = r_closed * np.cos(angle_closed)
y_contour = r_closed * np.sin(angle_closed)

# Plot
fig, ax = plt.subplots()
tcf = ax.tricontourf(x, y, I, levels=100, cmap='inferno')

ax.plot(x_contour, y_contour, color='cyan', linewidth=2, label='70% envelope')
ax.set_aspect('equal')
ax.set_title('Directional 70% Intensity Region')
ax.legend()
plt.colorbar(tcf, ax=ax, label='Intensity')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-1000,1000)
plt.ylim(-1000,1000)
plt.show()
"""