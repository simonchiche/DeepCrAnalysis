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
from scipy.interpolate import interp1d
import scipy
from scipy.interpolate import griddata
from datetime import datetime
from scipy.optimize import curve_fit
#endregion

#region Path definition
SimDir = "FullDenseDeepCr" #"DeepCrLibV1"  #"InterpSim"
WorkPath = os.getcwd()
BatchID = "GeomagneticAngle"
OutputPath = MatplotlibConfig(WorkPath, SimDir, BatchID)
#endregion
Save = True
simpath = "/Users/chiche/Desktop/DeepCrAnalysis/Simulations/" + SimDir
SimpathAll = glob.glob(simpath + "/*")

Eradair_allsims = []
Eradice_allsims = []
Eradtot = []

sin_alpha_all = []
theta_all = []
for simpath in SimpathAll[:1]:
    print(simpath.split("/")[-1])
    Shower = CreateShowerfromHDF5(simpath)
    uv = Shower.showerdirection()
    energy, theta, Nant = Shower.energy, Shower.zenith, Shower.nant
    if(energy!=0.316): continue
    B = Shower.B
    Bvec = [B[0], 0, -B[1]]
    uv[2] = -uv[2]
    ub = Bvec/np.linalg.norm(Bvec)
    alpha = np.arccos(np.dot(uv, ub))*180/np.pi
    print(alpha)
    sin_alpha_all.append(np.sin(alpha*np.pi/180))
    theta_all.append(theta)
    print(Shower.getXmaxPosition())

theta_all = np.array(theta_all)
sin_alpha_all = np.array(sin_alpha_all)
plt.figure()
plt.scatter(theta_all, sin_alpha_all, marker='x', color="teal", label='$sin{\\alpha}$', s=60)       
plt.xlabel("Zenith [Deg.]")
plt.ylabel("$sin{\\alpha}$")
plt.grid()
#plt.savefig(OutputPath + "_sin_alpha_vs_theta.pdf", bbox_inches = "tight") if Save else None
plt.show()
