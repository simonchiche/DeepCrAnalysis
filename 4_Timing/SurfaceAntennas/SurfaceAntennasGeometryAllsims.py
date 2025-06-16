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
import glob
from scipy.signal import hilbert
# endregion


#region Path definition
WorkPath = os.getcwd()
SimDir = "DeepCrLib"  
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
