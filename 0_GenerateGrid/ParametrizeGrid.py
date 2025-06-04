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


def generate_log_spaced_radial_grid(Lmax, Lcarac, N_target=1200, r_min=5):
    """
    Generate a 2D grid of positions (x, y) with radial symmetry.
    Radial spacing increases in log scale from r_min to Lmax.
    """

    # Create log-spaced radii from r_min to Lmax
    N_radii = int(np.logspace(np.log10(1), np.log10(50), num=20).sum())  # Estimate for loop
    radii = np.geomspace(r_min, Lmax, num=40)

    positions = []

    for r in radii:
        # Number of points on this ring ~ proportional to circumference
        # and inversely proportional to local spacing (to control density)
        n_points = int(2 * np.pi * r / (0.5 * r if r > Lcarac else 10))  # denser near center
        if n_points < 4:
            continue
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        for theta in angles:
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            positions.append((x, y))

    positions = np.array(positions)

    # Downsample to ~N_target if needed
    if len(positions) > N_target:
        idx = np.linspace(0, len(positions) - 1, N_target).astype(int)
        positions = positions[idx]

    return positions


positions = generate_log_spaced_radial_grid(
    Lmax=600,
    Lcarac=160,
    N_target=1200
)

plt.figure(figsize=(6,6))
plt.scatter(positions[:,0], positions[:,1], s=2)
plt.gca().set_aspect('equal')
#plt.title("Log-spaced radial antenna layout")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.grid(True)
plt.show()



xcarac = 160
xmax = 700
xlogmin = np.log(1)/np.log(xcarac)
xlogmax = np.log(xmax)/np.log(xcarac)

xloglin =  np.linspace(xlogmin, xlogmax, 32)

xlin = xcarac**xloglin
xlinsym = -xcarac**xloglin

angles = np.linspace(0,360,37)*np.pi/180.0

Nant = 1200
Nangles = len(angles)
Nradius = round(Nant/len(angles))
x_rad, y_rad = np.zeros(Nradius*Nangles), np.zeros(Nradius*Nangles)

for i in range(len(xlin)):

    for j in range(len(angles)):

        x_rad[Nangles*i + j] = xlin[i]*np.cos(angles[j])
        y_rad[Nangles*i + j] = xlin[i]*np.sin(angles[j])

plt.scatter(x_rad, y_rad)
plt.xlabel("x [m]")
plt.ylabel("y [m]")
#plt.xscale("symlog")
#plt.yscale("symlog")
plt.tight_layout()
plt.show()

xall = np.concatenate((xlin, xlinsym))
X, Y = np.meshgrid(xall, xall, indexing='ij') 

plt.scatter(X,Y)
#plt.xscale("symlog")
#plt.yscale("symlog")
plt.show()




xcore = np.linspace(5,200, 19)
xlogmin =  np.log(220)
xlogmax = np.log(800)
xloglin =  np.linspace(xlogmin, xlogmax, 20)
xlin = np.exp(xloglin)
xall = np.concatenate([xcore, xlin])
angles = np.linspace(0,360,37)*np.pi/180.0
Nangles = len(angles)
Nradius = len(xall)
Nant = Nradius*Nangles
x_rad, y_rad = np.zeros(Nradius*Nangles), np.zeros(Nradius*Nangles)

for i in range(len(xall)):

    for j in range(len(angles)):

        x_rad[Nangles*i + j] = xall[i]*np.cos(angles[j])
        y_rad[Nangles*i + j] = xall[i]*np.sin(angles[j])

plt.scatter(x_rad, y_rad)
plt.xlabel("x [m]")
plt.ylabel("y [m]")
#plt.xlim(-800, 800)
#plt.ylim(-800, 800)
#plt.xscale("symlog")
#plt.yscale("symlog")
plt.tight_layout()
plt.show()

def GenerateDenseCoreRadialArray(rmin, rcer, corestep, nrout, rmax):

   
    rcore = np.arange(rmin, rcer, corestep)
    rlogmin_out = np.log10(rcer)
    rlogmax_out = np.log10(rmax + 100)
    rlog_out =  np.linspace(rlogmin_out, rlogmax_out, nrout)

    rout = 10**rlog_out[1:]
    rall = np.concatenate([rcore, rout])
    angles = np.linspace(0, 2*np.pi, 36, endpoint=False)
    Nangles = len(angles)
    Nradius = len(rall)

    Nant = Nradius*Nangles
    x_rad, y_rad = np.zeros(Nradius*Nangles), np.zeros(Nradius*Nangles)

    for i in range(len(rall)):

        for j in range(len(angles)):

            x_rad[Nangles*i + j] = rall[i]*np.cos(angles[j])
            y_rad[Nangles*i + j] = rall[i]*np.sin(angles[j])
    
    return x_rad, y_rad

def GetExtent(Depth):

    if(Depth == 3216.0):
        extent = 300
    
    if(Depth == 3116.0):
        extent = 700
    
    else:
        extent = 500
    
    return extent

        

def GenerateDenseCoreArrayAllDepths(rmin, rcer, corestep, nrout, Depths):

    xrad_all, yrad_all, zrad_all = np.array([]), np.array([]),  np.array([])

    for i in range(len(Depths)):
        rmax =GetExtent(Depths[i])

        x_rad, y_rad = GenerateDenseCoreRadialArray(rmin, rcer, corestep, nrout, rmax)
        z_rad = np.full_like(x_rad, Depths[i])

        xrad_all = np.concatenate([xrad_all, x_rad])
        yrad_all = np.concatenate([yrad_all, y_rad])
        zrad_all = np.concatenate([zrad_all, z_rad])
        
    
    NantLay = len(x_rad)
    
    return xrad_all, yrad_all, zrad_all, NantLay

selDepth = np.array([3216, 3156, 3116])

x_rad, y_rad, z_rad, NantLay = GenerateDenseCoreArrayAllDepths(5, 200, 10, 20, selDepth)


plt.scatter(x_rad[:NantLay], y_rad[:NantLay])

plt.scatter(x_rad, y_rad, s =1)
plt.xlabel("x [m]")
plt.ylabel("y [m]")
#plt.xlim(-800, 800)
#plt.ylim(-800, 800)
#plt.xscale("symlog")
#plt.yscale("symlog")
plt.tight_layout()
plt.show()