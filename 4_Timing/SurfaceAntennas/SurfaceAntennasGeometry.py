#region Modules 
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os
from Modules.ModuleSurfaceAntennas import cartesian_to_spherical_angles, GetShowerDirection, generate_footprint, PropagateRayAll, sample_points_in_polygon, getXmaxPosition, GetDantXmax, GetTransmittedFraction
from Modules.ModulePlotSurfaceAntennas import PlotSurfaceFootprint, CompareFootprints, PlotFootprintPolygons, PlotSampledFootprint, PlotTimeDelayDistribution, PlotAmplitudeDilution
from  MainModules.PlotConfig import MatplotlibConfig
from MainModules.ShowerClass import CreateShowerfromHDF5
from shapely.geometry import Polygon
# endregion


#region Path definition
WorkPath = os.getcwd()
SimDir = "DeepCrLib"  
SimName = "Rectangle_Proton_0.316_0_0_1"
simpath = "/Users/chiche/Desktop/DeepCrAnalysis/Simulations/DeepCrLibV1/"\
      + SimName + "_0.hdf5"
BatchID = "IndividualSufaceAntennas"
OutputPath = MatplotlibConfig(WorkPath, SimDir, BatchID)
# endregion
Save = False

# region Shower parameters
Shower = CreateShowerfromHDF5(simpath)
XmaxPos= Shower.xmaxpos
if(Shower.zenith == 0): Shower.zenith = 0.01 # avoid division by zero in the footprint generation
if(Shower.xmaxdist==0): 
    Shower.xmaxdist = 1 # avoid division by zero in the footprint generation
    XmaxPos = Shower.getXmaxPosition()
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

# Sampled positions for time and amplitude distributions
Nsamples = 1000
footprint_samples = sample_points_in_polygon(footprint, Nsamples)

all_xray_samples, all_yray_samples, all_zray_samples, all_nray_samples, all_dt_samples, all_dL_samples =\
    PropagateRayAll(footprint_samples, XmaxPos, 100, Shower.glevel, IceModel)

# Amplitude dilution calculation
DantXmax =GetDantXmax(footprint_samples, XmaxPos, Shower)
DilutionFactor = DantXmax/(DantXmax + np.array(all_dL_samples))

# Reflection coefficient
TransFrac_ortho = GetTransmittedFraction(XmaxPos, footprint_samples, IceModel, Shower)

# Cummulative effects of dilution and transmission fraction
SurfaceDeepRatio = DilutionFactor*TransFrac_ortho


#######  PLOTS #########
# Display the surface footprint
Save = True
# Plot the surface footprint
PlotSurfaceFootprint(footprint, Shower, Save, BatchID, OutputPath)
# Compare the surface footprint with the in-ice footprint
CompareFootprints(footprint, Shower, all_xray, all_yray, Save, BatchID, OutputPath)

# Plot footprint polygons with deep trigger only fraction
PlotFootprintPolygons(polygon_surface, polygon_deep, Shower, Save, BatchID, OutputPath)

# Sampled surface footprint
PlotSampledFootprint(footprint_samples, Shower, Save, BatchID, OutputPath)

# Histogram of the time delay distribution
PlotTimeDelayDistribution(all_dt_samples, Shower, Save, BatchID, OutputPath)

# Amplitude dilution scatter plot
PlotAmplitudeDilution(all_xray_samples, all_yray_samples, SurfaceDeepRatio,  Shower, Save, BatchID, OutputPath)


