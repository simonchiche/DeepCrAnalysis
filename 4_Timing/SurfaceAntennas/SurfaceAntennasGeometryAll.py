#region Modules 
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os
from Modules.ModuleSurfaceAntennas import cartesian_to_spherical_angles, GetShowerDirection, generate_footprint, PropagateRayAll, sample_points_in_polygon, getXmaxPosition, GetDantXmax, GetTransmittedFraction, GetDeepTriggerFrac
from Modules.ModulePlotSurfaceAntennas import PlotSurfaceFootprint, CompareFootprints, PlotFootprintPolygons, PlotSampledFootprint, PlotTimeDelayDistribution, PlotAmplitudeDilution
from  MainModules.PlotConfig import MatplotlibConfig
from MainModules.ShowerClass import CreateShowerfromHDF5
from shapely.geometry import Polygon
import glob
# endregion


#region Path definition
WorkPath = os.getcwd()
SimDir = "DeepCrLib"  
SimName = "Rectangle_Proton_0.0316_0_0_1"
simpath = "/Users/chiche/Desktop/DeepCrAnalysis/Simulations/DeepCrLibV1/"\
      + SimName + "_0.hdf5"
BatchID = "IndividualSufaceAntennas"
OutputPath = MatplotlibConfig(WorkPath, SimDir, BatchID)
Save = False
SimpathAll = glob.glob("/Users/chiche/Desktop/DeepCrAnalysis/Simulations/DeepCrLibV1/*")

EnergyAll, ZenithAll, XmaxDistAll, DeepTriggerAll, dt_all_sims = \
    ([] for _ in range(5))

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


    # Surface footprint contours
    footprint = \
        generate_footprint(Shower, theta_lim, 360)

    # In-ice footprint contours
    all_xray, all_yray, all_zray, all_nray, all_dt, all_dL =\
        PropagateRayAll(footprint, XmaxPos, 100, Shower.glevel, IceModel)

    # Footpint comparison and deep triggers only area
    polygon_surface = Polygon(footprint)
    ice_footprint_coords = np.column_stack((all_xray, all_yray))
    polygon_deep = Polygon(ice_footprint_coords)
    DeepTriggerFrac = GetDeepTriggerFrac(polygon_surface, polygon_deep)
    DeepTriggerAll.append(DeepTriggerFrac)

    # Sampled positions for time and amplitude distributions
    Nsamples = 1000
    footprint_samples = sample_points_in_polygon(footprint, Nsamples)

    all_xray_samples, all_yray_samples, all_zray_samples, all_nray_samples, all_dt_samples, all_dL_samples =\
        PropagateRayAll(footprint_samples, XmaxPos, 100, Shower.glevel, IceModel)
    
    dt_all_sims.append(all_dt_samples)

    # Amplitude dilution calculation
    DantXmax =GetDantXmax(footprint_samples, XmaxPos, Shower)
    DilutionFactor = DantXmax/(DantXmax + np.array(all_dL_samples))

    # Reflection coefficient
    TransFrac_ortho = GetTransmittedFraction(XmaxPos, footprint_samples, IceModel, Shower)

    # Cummulative effects of dilution and transmission fraction
    SurfaceDeepRatio = DilutionFactor*TransFrac_ortho


    #######  PLOTS #########
    # Display the surface footprint
    Save = False
    # Plot the surface footprint
    ##PlotSurfaceFootprint(footprint, Shower, Save, BatchID, OutputPath)
    # Compare the surface footprint with the in-ice footprint
    ##CompareFootprints(footprint, Shower, all_xray, all_yray, Save, BatchID, OutputPath)

    # Plot footprint polygons with deep trigger only fraction
    ##PlotFootprintPolygons(polygon_surface, polygon_deep, Shower, Save, BatchID, OutputPath)

    # Sampled surface footprint
    ##PlotSampledFootprint(footprint_samples, Shower, Save, BatchID, OutputPath)

    # Histogram of the time delay distribution
    ##PlotTimeDelayDistribution(all_dt_samples, Shower, Save, BatchID, OutputPath)

    # Amplitude dilution scatter plot
    ##PlotAmplitudeDilution(all_xray_samples, all_yray_samples, SurfaceDeepRatio,  Shower, Save, BatchID, OutputPath)


EnergyAll, ZenithAll, XmaxDistAll, DeepTriggerAll, dt_all_sims =\
    map(np.array, [EnergyAll, ZenithAll, XmaxDistAll, DeepTriggerAll, dt_all_sims])


def PlotXmaxDistanceVsZenith(EnergyAll, ZenithAll, XmaxDistAll):
    """
    Plot the Xmax distance as a function of the zenith angle for different energies.
    """
    plt.figure(figsize=(10, 6))
    Ebins = np.unique(EnergyAll)
    for i in range(len(Ebins)):
        sel = EnergyAll == Ebins[i]
        arg = np.argsort(ZenithAll[sel])
        plt.plot(ZenithAll[sel][arg], XmaxDistAll[sel][arg]/1e3, '-o', label=f"E={Ebins[i]} EeV")
    plt.xlabel("$\\theta$ [degrees]")
    plt.ylabel("Xmax distance [km]")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    #plt.yscale('log')
    plt.show()


def PlotDeepTriggersVsZenith(EnergyAll, ZenithAll, DeepTriggerAll):
    """
    Plot the deep trigger fraction as a function of the zenith angle for different energies.
    """
    #plt.figure(figsize=(10, 6))
    Ebins = np.unique(EnergyAll)
    for i in range(len(Ebins)):
        sel = EnergyAll == Ebins[i]
        arg = np.argsort(ZenithAll[sel])
        plt.plot(ZenithAll[sel][arg], DeepTriggerAll[sel][arg]*100, 'o', label=f"E={Ebins[i]} EeV")
    plt.xlabel("$\\theta$ [degrees]")
    plt.ylabel("Deep trigger only events [$\%$]")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    #plt.yscale('log')
    plt.show()

PlotXmaxDistanceVsZenith(EnergyAll, ZenithAll, XmaxDistAll)
PlotDeepTriggersVsZenith(EnergyAll, ZenithAll, DeepTriggerAll)

ZenithBins = np.unique(ZenithAll)
zenithmask = np.array([10, 28, 39, 50])
selE = 0.316

bins = np.linspace(500, 750, 50 + 1)

argsort = np.argsort(ZenithAll)

for i in range(len(ZenithAll[argsort])):
    if(ZenithAll[argsort][i] not in zenithmask): continue
    if(EnergyAll[argsort][i] != selE): continue
    print(i)
    plt.hist(dt_all_sims[argsort][i]*1e9, bins = bins, alpha=0.7, edgecolor='black', label=f"$\\theta$={ZenithAll[argsort][i]}$^\\circ$")
plt.xlabel("Time delay [ns]")
plt.ylabel("Count")
plt.legend()
plt.show()


plt.hist(dt_all_sims[16])

for i in range(len(ZenithBins)):
    if(ZenithBins[i] not in zenithmask): continue
    plt.hist(dt_all_sims[i]*1e9, bins = 20, alpha=0.7, edgecolor='black', label=f"$\\theta$={ZenithBins[i]} degrees")
plt.xlabel("Time delay [ns]")
plt.ylabel("Count")
plt.legend()
plt.show()

#plt.xlabel("Energy [EeV]")

#plt.hist(dt_all_sims, bins = 20, color='blue', alpha=0.7, edgecolor='black')