#region Modules 
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os
from Modules.ModuleSurfaceAntennas import cartesian_to_spherical_angles, GetShowerDirection, generate_footprint, PropagateRayAll, sample_points_in_polygon, getXmaxPosition, GetDantXmax, GetTransmittedFraction, GetDeepTriggerFrac, GetDeltatDistributionfromSims, GetMeandeltat
from Modules.ModulePlotSurfaceAntennas import PlotSurfaceFootprint, CompareFootprints, PlotFootprintPolygons, PlotSampledFootprint, PlotTimeDelayDistribution, PlotAmplitudeDilution, PlotXmaxDistanceVsZenith, PlotTimeDistributionAllsimsperEbin, PlotMeanTimedelayEbin
from  MainModules.PlotConfig import MatplotlibConfig
from MainModules.ShowerClass import CreateShowerfromHDF5
from shapely.geometry import Polygon
from Modules.ModuleSurfaceAntennas import n
import glob
from scipy.signal import hilbert
# endregion


#region Path definition
WorkPath = os.getcwd()
SimDir = "DeepCrLibV1"  
SimName = "Rectangle_Proton_0.0316_0_0_1"
simpath = "/Users/chiche/Desktop/DeepCrAnalysis/Simulations/DeepCrLibV1/"\
      + SimName + "_0.hdf5"
BatchID = "SimulationSurfaceAntennas"
OutputPath = MatplotlibConfig(WorkPath, SimDir, BatchID)
Save = False
SimpathAll = glob.glob("/Users/chiche/Desktop/DeepCrAnalysis/Simulations/DeepCrLibV1/*")

EnergyAll, ZenithAll, XmaxDistAll, dt_all_sims = \
    ([] for _ in range(4))

for simpath in SimpathAll:
    Shower = CreateShowerfromHDF5(simpath)

    XmaxPos= Shower.xmaxpos
    if(Shower.zenith == 0): continue #Shower.zenith = 0.01 # avoid division by zero in the footprint generation
    if(Shower.xmaxdist==0): 
        Shower.xmaxdist = 1 # avoid division by zero in the footprint generation
        XmaxPos = Shower.getXmaxPosition()
    EnergyAll.append(Shower.energy)
    ZenithAll.append(Shower.zenith)
    XmaxDistAll.append(Shower.xmaxdist)
    # Cerenkov angle in degrees
    theta_C = 1.2 
    # footprint aperture angle in degrees
    theta_lim = 3*theta_C  
    IceModel = 1 # Greenland ice model
    # endregion 

    deltat, tsurface, tdeep, Pos_surface, Pos_deep = \
        GetDeltatDistributionfromSims(Shower, 100)
    dt_all_sims.append(deltat)


    Save = False
    # Histogram of the time delay distribution
    PlotTimeDelayDistribution(deltat, Shower, Save, BatchID, OutputPath)

EnergyAll, ZenithAll, XmaxDistAll =\
    map(np.array, [EnergyAll, ZenithAll, XmaxDistAll])

mean_deltat, std_deltat = GetMeandeltat(dt_all_sims)


#######  PLOTS #########
PlotXmaxDistanceVsZenith(EnergyAll, ZenithAll, XmaxDistAll)
PlotTimeDistributionAllsimsperEbin(ZenithAll, EnergyAll, dt_all_sims, selE=0.1)
PlotMeanTimedelayEbin(ZenithAll, EnergyAll, mean_deltat, std_deltat)


# Simulation results only
Ebins = np.unique(EnergyAll)
for i in range(len(Ebins)):
    sel2  = EnergyAll == Ebins[i]
    arg2 = np.argsort(ZenithAll[sel2])
    plt.errorbar(ZenithAll[sel2][arg2], mean_deltat[sel2][arg2]*1e9, yerr=std_deltat[sel2][arg2]*1e9, fmt='o', label=f"E={Ebins[i]} EeV")
    plt.ylabel("Time delay [ns]")
    plt.xlabel("zenith [Deg.]")
    plt.legend()
    plt.grid()
    plt.show()

# Analytical results only
ZenithAll_analytic, EnergyAll_analytic, mean_deltat_analytic, std_deltat_analytic = np.loadtxt("./Data/TimeDelay.txt", unpack=True).T
PlotMeanTimedelayEbin(ZenithAll_analytic, EnergyAll_analytic, mean_deltat_analytic, std_deltat_analytic)

Ebins = np.unique(EnergyAll)
for i in range(len(Ebins)):
    #sel = EnergyAll == Ebins[i]
    sel2 = EnergyAll_analytic == Ebins[i]
    #arg = np.argsort(ZenithAll[sel])
    arg2 = np.argsort(ZenithAll_analytic[sel2])
    #plt.errorbar(ZenithAll[sel][arg], mean_deltat[sel][arg]*1e9, yerr=std_deltat[sel][arg]*1e9, fmt='o', label=f"E={Ebins[i]} EeV")
    plt.errorbar(ZenithAll_analytic[sel2][arg2], mean_deltat_analytic[sel2][arg2]*1e9, yerr=std_deltat_analytic[sel2][arg2]*1e9, fmt='o', label=f"E={Ebins[i]} EeV")
    plt.ylabel("Time delay [ns]")
    plt.xlabel("zenith [Deg.]")
    plt.legend()
    plt.show()


# Simulation vs analytic results
Ebins = np.unique(EnergyAll_analytic)
for i in range(len(Ebins)):
    sel = EnergyAll_analytic == Ebins[i]
    sel2  = EnergyAll == Ebins[i]
    arg = np.argsort(ZenithAll_analytic[sel])
    arg2 = np.argsort(ZenithAll[sel2])
    plt.errorbar(ZenithAll[sel2][arg2], mean_deltat[sel2][arg2]*1e9, yerr=std_deltat[sel2][arg2]*1e9, fmt='o', label="Simulations")
    plt.errorbar(ZenithAll[sel][arg], mean_deltat_analytic[sel][arg]*1e9, yerr=std_deltat_analytic[sel][arg]*1e9, fmt='o', label="Analytic")
    plt.ylabel("Time delay [ns]")
    plt.xlabel("zenith [Deg.]")
    plt.legend()
    plt.show()

# Simulation vs analytic results without error bars
Ebins = np.unique(EnergyAll_analytic)
for i in range(len(Ebins)):
    sel = EnergyAll_analytic == Ebins[i]
    sel2  = EnergyAll == Ebins[i]
    arg = np.argsort(ZenithAll_analytic[sel])
    arg2 = np.argsort(ZenithAll[sel2])
    plt.scatter(ZenithAll[sel2][arg2],  mean_deltat[sel2][arg2]*1e9, label=f"E={Ebins[i]} EeV")
    plt.scatter(ZenithAll[sel][arg], mean_deltat_analytic[sel][arg]*1e9, label=f"E={Ebins[i]} EeV")
    plt.ylabel("Time delay [ns]")
    plt.xlabel("zenith [Deg.]")
    plt.legend()
    plt.show()

# All zenith distributions at E = 0.316 EeV
dt_all_sims_analytic= np.loadtxt("./Data/TimeDelayDistrib.txt")
dtflat17_5_sim = []
for i in range(len(EnergyAll)):
    if(EnergyAll[i] == 0.316):
        dtflat17_5_sim.append(dt_all_sims[i][:])

dtflat17_5_analytic = []
for i in range(len(EnergyAll_analytic)):
    if(EnergyAll_analytic[i] == 0.316):
        dtflat17_5_analytic.append(dt_all_sims_analytic[i][:])


Deltat_distrib_sim_E17_5_AllZeniths = [item for sublist in dtflat17_5_sim for item in sublist]
plt.hist(Deltat_distrib_sim_E17_5_AllZeniths, bins=10, label='Simulated', edgecolor='black', alpha=0.7)

Deltat_distrib_analytic_E17_5_AllZeniths = [item for sublist in dtflat17_5_analytic for item in sublist]

bins = np.linspace(350, 750, 30 + 1)/1e9
plt.hist(Deltat_distrib_sim_E17_5_AllZeniths, edgecolor='black', alpha=0.7, bins=bins, density=True, label ="simulation")
plt.hist(Deltat_distrib_analytic_E17_5_AllZeniths, edgecolor='black', alpha=0.7, bins=bins, density=True,  label ="analytic")
plt.xlabel("Time [s]")
plt.ylabel("Counts")
plt.legend()
plt.show()




# Plots for the paper

# Simulation results only
selE = 0.316
Ebins = np.unique(EnergyAll)
sel2  = EnergyAll ==selE
arg2 = np.argsort(ZenithAll[sel2])
plt.errorbar(ZenithAll[sel2][arg2], mean_deltat[sel2][arg2]*1e9, yerr=std_deltat[sel2][arg2]*1e9, fmt='o', label=f"E={Ebins[i]} EeV", color="#ff7f00")
plt.ylabel("Time delay [ns]")
plt.xlabel("zenith [Deg.]")
plt.legend()
plt.grid()
plt.ylim(450, 700)
#plt.savefig(OutputPath + "_MeanTimeDelay_vs_Zenith_E%.3f.pdf" %selE, bbox_inches = "tight")
plt.show()

# Calculate the ice mean index of refraction
zdepths = np.linspace(0,-100, 1000)
ice_index = n(zdepths[1], 1)
ice_index_all =  np.zeros(len(zdepths))
for i in range(len(zdepths)):
    ice_index_all[i] = n(zdepths[i], 1) 
plt.plot(zdepths, ice_index_all, label="Ice index of refraction")
print(np.mean(ice_index_all))
# Derive the corresponding time delay in ns
dt_ice_model = (100*1.605/3e8)*1e9
print(dt_ice_model)

weights = np.ones_like(Deltat_distrib_sim_E17_5_AllZeniths) / len(Deltat_distrib_sim_E17_5_AllZeniths) * 100
bins = np.linspace(350, 750, 30 + 1)
plt.hist(np.array(Deltat_distrib_sim_E17_5_AllZeniths)*1e9,weights=weights, label='$\Delta t$', edgecolor='black', alpha=0.7, bins=bins, color='#377eb8')
#plt.axvline(dt_ice_model, color='#e41a1c', linestyle='--', label='$\\bar{n} \Delta x/c$', linewidth=2)
plt.xlabel("Time delay [ns]")
plt.ylabel("Antennas perecentage [%]")
plt.legend()
#plt.savefig(OutputPath + "_TimeDelayDistrib_E%.3f.pdf" %selE, bbox_inches = "tight")
plt.show()



