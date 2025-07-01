#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 02:21:35 2024

@author: chiche
"""


import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
import glob
import sys
import pickle
#from Modules.SimParam.GetLayoutScaling import GetDepthScaling, GetEnergyScaling
from scipy.interpolate import interp1d
import scipy
from Modules.FunctionsGetFluence import  Norm, LoadTraces, GetPeakTraces, Traces_cgs_to_si, GetDepths, CorrectScaling, CombineTraces, CorrectLength, GetIntTraces, GetIntTracesSum, GetRadioExtent
from Modules.FunctionsPlotFluence import EfieldMap, PlotLDF, PlotTraces, plot_polarisation, PlotMaxTraces, PlotAllTraces, PlotLayer, PlotGivenTrace, PlotAllChannels
from scipy.interpolate import griddata
from datetime import datetime
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt
from Modules.FunctionsPlotFluence import EfieldMap, PlotLDF, PlotTraces, plot_polarisation, PlotMaxTraces, PlotAllTraces, PlotLayer, PlotGivenTrace, PlotAllChannels, PlotSurfaceEz, RemoveCoreAntennas, InterpolatedEfieldMap


PowerDataPath = "/Users/chiche/Desktop/DeepCrSearch/Analysis/GetFluence/Data/Power/"

SimDir =     "InterpSim" #"InterpSim"
SimName = "Rectangle_Proton_0.0316_0_0_1"


font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 14}

plt.rc('font', **font)
zenith = float(SimName.split("_")[3])
Save = False

# =============================================================================
#                              Load Traces
# =============================================================================

Path =  "/Users/chiche/Desktop/DeepCrSearch"\
+ "/Simulations/" + SimDir + "/" + SimName + "/"
energy = float(Path.split("/")[-2].split("_")[2])
theta  = float(Path.split("/")[-2].split("_")[3])

SimDataPath = "/Users/chiche/Desktop/DeepCrSearch/Analysis/SimData/" + SimDir + "/" + SimName 

if(not(os.path.exists(SimDataPath))):
    
    cmd = "mkdir -p " + SimDataPath
    p =subprocess.Popen(cmd, cwd=os.getcwd(), shell=True)
    stdout, stderr = p.communicate()
    Nant, Traces_C, Traces_G, Pos = LoadTraces(Path)
    
    np.save(SimDataPath + "/Nant", Nant)
    with open(SimDataPath + '/Traces_C.pkl', 'wb') as file:
        pickle.dump(Traces_C, file)
    with open(SimDataPath + '/Traces_G.pkl', 'wb') as file:
        pickle.dump(Traces_G, file)
    np.save(SimDataPath + "/Pos", Pos)
    
else:
    
    Nant  = np.load(SimDataPath + "/Nant.npy")
    with open(SimDataPath + '/Traces_C.pkl', 'rb') as file:
        Traces_C = pickle.load(file)
    with open(SimDataPath + '/Traces_G.pkl', 'rb') as file:
        Traces_G = pickle.load(file)
    Pos  = np.load(SimDataPath + "/Pos.npy", allow_pickle=True)
        
    
Nlay, Nplane, Depths = GetDepths(Pos)

# To resize the length of the Coreas traces
CorrectLength(Traces_C, False)

#cgs to si
Traces_C = Traces_cgs_to_si(Traces_C)
Traces_G = Traces_cgs_to_si(Traces_G)


# =============================================================================
#                           Coherent sum
# =============================================================================

if(not(os.path.exists(SimDataPath + "/Traces_tot.pkl"))):
    Traces_tot = CombineTraces(Nant, Traces_C, Traces_G)
    with open(SimDataPath + '/Traces_tot.pkl', 'wb') as file:
        pickle.dump(Traces_C, file)   
else:
    with open(SimDataPath + '/Traces_tot.pkl', 'rb') as file:
        Traces_tot = pickle.load(file)    

# =============================================================================
#                           Get peak amplitude
# =============================================================================
# Peak value of the traces
ExC, EyC, EzC, EtotC = GetPeakTraces(Traces_C, Nant)
ExG, EyG, EzG, EtotG = GetPeakTraces(Traces_G, Nant)
Extot, Eytot, Eztot, Etot_peak = GetPeakTraces(Traces_tot, Nant)

# =============================================================================
#                                 Get integral
# =============================================================================

ExC_int, EyC_int, EzC_int, EtotC_int = GetIntTraces(Traces_C, Nant)

ExG_int, EyG_int, EzG_int, EtotG_int = GetIntTraces(Traces_G, Nant)

Ex_tot_int, Ey_tot_int, Ez_tot_int, Etot_int = GetIntTraces(Traces_tot, Nant)





def bandpass_filter(signal, fs, lowcut, highcut, order=4):
    """
    Apply a bandpass filter to a signal.
    
    Parameters:
    - signal: array-like, the input signal (E(t)).
    - fs: float, the sampling frequency of the signal in Hz.
    - lowcut: float, the lower bound of the frequency band in Hz.
    - highcut: float, the upper bound of the frequency band in Hz.
    - order: int, the order of the Butterworth filter (default is 4).
    
    Returns:
    - filtered_signal: array-like, the filtered signal.
    """
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist

    # Design the Butterworth bandpass filter
    b, a = butter(order, [low, high], btype='band')

    # Apply the filter using filtfilt for zero phase shift
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

from scipy.interpolate import Rbf
from scipy.interpolate import griddata

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


fs = 5e9
lowcut = 10e6  # Lower bound of the frequency band in Hz (50 MHz)
highcut = 1e9  # Upper bound of the frequency band in Hz (2000 MHz)

Eg_f = np.zeros(len(Traces_G))
filtered_signal = []
for i in range(len(Traces_G)):
    Exg_f = bandpass_filter(Traces_G[i][:,1], fs, lowcut, highcut)
    Eyg_f = bandpass_filter(Traces_G[i][:,2], fs, lowcut, highcut)
    Ezg_f = bandpass_filter(Traces_G[i][:,3], fs, lowcut, highcut)
    
    Etotg_f = np.sqrt(Exg_f**2 + Eyg_f**2 + Ezg_f**2)
    #filtered_signal.append(bandpass_filter(Traces_G[i][:,2], fs, lowcut, highcut))
    Eg_f[i] = max(abs(Etotg_f))

    


# Geant 
#EfieldMap(Pos, Nlay, Nplane, Eg_f, "GeantHilbert",\
#          True, energy, theta, OutputPath)

Save = False
E = Eg_f
Nlay = len(Depths)
for i in range(Nlay):
    sel = (Pos[:,2] == Depths[i])
    #s = 10 + 20 * (E[sel] - np.min(E)) / (np.max(E) - np.min(E))
    plt.scatter(Pos[sel,0], Pos[sel,1], \
                c= E[sel], cmap = "jet", edgecolors='k', linewidth=0.)
    cbar = plt.colorbar()
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    cbar.set_label("$\log_{10}(E)$ [$\mu V/m$]")
    depth =Depths[0]- Depths[i]
    #plt.xlim(-200,200)
    #plt.ylim(-200,200)
    plt.legend(["Depth = %.f m" %(depth)], loc ="upper right")
    plt.title("(E =$%.2f\,$EeV, $\\theta=%.1f^{\circ}$)" %(energy, theta), size =12)
    plt.grid(True, linestyle='--', alpha=0.3)
    if(Save):
        plt.savefig\
        ("/Users/chiche/Desktop/" + "EfieldMap_E%.2f_th%.1f_depth%1.f.pdf" \
            %(energy, theta, depth), bbox_inches = "tight")
    plt.show()

Save = True

#Eg_f[Eg_f > 400] = np.max(Eg_f)
Nlay = len(Depths)
for i in range(Nlay):
    sel = (Pos[:,2] == Depths[i])
    #
    # s = 10 + 20 * (E[sel] - np.min(E)) / (np.max(E) - np.min(E))
    grid_x, grid_y, grid_z = \
        interpolate_rbf(Pos[:,0][sel], Pos[:,1][sel], E[sel]+1)
    
    plt.figure(figsize=(6, 5))
    plt.contourf(grid_x, grid_y, grid_z, levels=100, cmap='jet')
    #plt.scatter(Pos[:729,0], Pos[:729,1], c=np.log10(EtotC_int[:729] +1), edgecolor='white', s=100)
    #plt.colorbar(label="$\log_{10}(E)$ [$\mu Vs/m$]")
    plt.colorbar(label="E [$\mu V/m$]")
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    depth =Depths[0]- Depths[i]
    plt.title("In-ice map (E =$10^{16.5}\,$eV, $\\theta=%.1f^{\circ}$)" %(theta), size =12)
    plt.text(0.05, 0.95, f"Depth = 100 m", transform=plt.gca().transAxes,
                 fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
    if(Save):
        plt.savefig\
        ("/Users/chiche/Desktop/" + "EfieldMap_E%.2f_th%.1f_depth%1.f.pdf" \
            %(energy, theta, depth), bbox_inches = "tight")
    #plt.legend(["Depth = %.f m" %(depth)], loc ="upper right")
    #plt.text(0.05, 0.95, f"Depth = {depth:.0f} m", transform=plt.gca().transAxes,
    #            fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
    #plt.xlim(min(grid_x), max(grid_x))
    #plt.ylim(min(grid_x), max(grid_x))
    
    plt.show()