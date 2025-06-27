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
#sys.path.append("/Users/chiche/Desktop/DeepCrSearch/Analysis/")
from Modules.SimParam.GetLayoutScaling import GetDepthScaling, GetEnergyScaling
from scipy.interpolate import interp1d
import scipy
from Modules.Fluence.FunctionsGetFluence import  Norm, LoadTraces, GetPeakTraces, Traces_cgs_to_si, GetDepths, CorrectScaling, CombineTraces, CorrectLength, GetIntTraces, GetIntTracesSum, GetRadioExtent
from Modules.Fluence.FunctionsPlotFluence import EfieldMap, PlotLDF, PlotTraces, plot_polarisation, PlotMaxTraces, PlotAllTraces, PlotLayer, PlotGivenTrace, PlotAllChannels
from CleanCoreasTraces import CleanCoreasTraces
from Modules.SimParam.PlotRadioSimExtent import PlotFillingFactor, PlotRadioSimExtent
from scipy.interpolate import griddata
from datetime import datetime
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt
from Modules.Fluence.FunctionsRadiationEnergy import GetFluence, GetRadiationEnergy
from ModulePlotDumbleBumps import PlotPeakEfield, PlotDumbleBumpsMaps, PlotDoubleBumpVsZen, PlotTimeDelay
from ModuleDoubleBumps import LoadSimulation, GetDoubleBumps

PowerDataPath = "/Users/chiche/Desktop/DeepCrSearch/Analysis/GetFluence/Data/Power/"

SimDir = "DeepCrLib"  #"InterpSim"
SimName = "Rectangle_Proton_0.316_0_0_1"

# We create a directory where the outputs are stored
date = datetime.today().strftime('%Y-%m-%d')
WorkPath = os.getcwd()
OutputPath = WorkPath + "/Plots/" + SimDir + "/" + date + "/" 
#print(OutputPath)
#sys.exit()
cmd = "mkdir -p " + OutputPath
p =subprocess.Popen(cmd, cwd=os.getcwd(), shell=True)
stdout, stderr = p.communicate()

font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 14}

plt.rc('font', **font)
zenith = float(SimName.split("_")[3])
Save = False
savedir = "/Users/chiche/Desktop/RNO_Brussels_20_02_25"
# =============================================================================
#                              Load Traces
# =============================================================================

Path =  "/Users/chiche/Desktop/DeepCrSearch"\
+ "/Simulations/" + SimDir + "/" + SimName + "/"
energy = float(Path.split("/")[-2].split("_")[2])
theta  = float(Path.split("/")[-2].split("_")[3])

SimDataPath = \
"/Users/chiche/Desktop/DeepCrSearch/Analysis/SimData/" + SimDir + "/" + SimName 
Nant, Traces_C, Traces_G, Pos = LoadSimulation(SimDataPath)    
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

NantLay = 729
EtotC100, EtotG100, isDoubleBump, isSingleBump, DeltaT = \
      GetDoubleBumps(NantLay, Traces_C, Traces_G, Traces_tot, thresold=40, Plot = False)

print(np.sum(isSingleBump))

# Peak of EtotC
PlotPeakEfield(Pos, np.log(EtotC100), NantLay, energy, zenith)

# Peak of EtotG
PlotPeakEfield(Pos, np.log(EtotG100+1), NantLay, energy, zenith)
PlotPeakEfield(Pos, EtotG100, NantLay, energy, zenith)

# Double Bump maps
PlotDumbleBumpsMaps(Pos, isDoubleBump, NantLay, energy, zenith)


#Deltat = Tmaxg - Tmaxc
#plt.plot(Deltat)
#plt.show()
Ndouble = len(isDoubleBump[isDoubleBump==1])
fname = "/DoublePulseAll.txt"

DeltaTmean = np.mean(DeltaT)
#with open(savedir + fname, "a") as f1:
#    f1.write(f"{energy}\t{zenith}\t{Ndouble}\t{DeltaTmean}\n")

Eall, ZenAll, NdoubleAll, DeltaTmean = np.loadtxt(savedir + fname, unpack=True)
PlotDoubleBumpVsZen(ZenAll, NdoubleAll)
PlotTimeDelay(DeltaT)


#--------------------------------------------------------------
plt.scatter(Eall, DeltaTmean*1e9)
plt.grid()
plt.xlabel("E [Eev]")
plt.ylabel("$<\Delta t>$ [ns]")
#plt.savefig("/Users/chiche/Desktop/MeanDoublePulseDelay_vs_E.pdf", bbox_inches = "tight")
plt.show()


plt.scatter(ZenAll[Eall == 0.0316], DeltaTmean[Eall == 0.0316]*1e9, label = "E  = $10^{16.5}$ EeV")
plt.scatter(ZenAll[Eall == 0.1], DeltaTmean[Eall == 0.1]*1e9, label = "E  = $10^{17}$ EeV")
plt.scatter(ZenAll[Eall == 0.316], DeltaTmean[Eall == 0.316]*1e9, label = "E  = $10^{17.5}$ EeV")
plt.grid()
plt.xlabel("E [Eev]")
plt.ylabel("$<\Delta t>$ [ns]")
plt.legend()
#plt.savefig("/Users/chiche/Desktop/MeanDoublePulseDelay_vs_theta.pdf", bbox_inches = "tight")
plt.show()


plt.scatter(ZenAll[Eall == 0.0316], NdoubleAll[Eall == 0.0316], label = "E  = $10^{16.5}$ EeV")
plt.scatter(ZenAll[Eall == 0.1], NdoubleAll[Eall == 0.1], label = "E  = $10^{17}$ EeV")
plt.scatter(ZenAll[Eall == 0.316], NdoubleAll[Eall == 0.316], label = "E  = $10^{17.5}$ EeV")
plt.grid()
plt.xlabel("E [Eev]")
plt.ylabel("$N_{double}$")
plt.legend()
#plt.savefig("/Users/chiche/Desktop/NdoublePulse_vs_theta.pdf", bbox_inches = "tight")
plt.show()


plt.scatter(Eall[Eall == 0.0316], NdoubleAll[Eall == 0.0316], label = "E  = $10^{16.5}$ EeV")
plt.scatter(Eall[Eall == 0.1], NdoubleAll[Eall == 0.1], label = "E  = $10^{17}$ EeV")
plt.scatter(Eall[Eall == 0.316], NdoubleAll[Eall == 0.316], label = "E  = $10^{17.5}$ EeV")
plt.grid()
plt.xlabel("E [Eev]")
plt.ylabel("$N_{double}$")
plt.legend()
#plt.savefig("/Users/chiche/Desktop/NdoublePulse_vs_E.pdf", bbox_inches = "tight")
plt.show()