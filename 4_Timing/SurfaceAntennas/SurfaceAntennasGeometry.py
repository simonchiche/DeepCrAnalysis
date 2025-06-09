#region Modules 
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os
from Modules.ModuleSurfaceAntennas import cartesian_to_spherical_angles, GetShowerDirection, generate_footprint, PropagateRayAll, sample_points_in_polygon, getXmaxPosition, GetDantXmax, GetTransmittedFraction
from Modules.ModulePlotSurfaceAntennas import PlotSurfaceFootprint, CompareFootprints
from  MainModules.PlotConfig import MatplotlibConfig
from MainModules.ShowerClass import CreateShowerfromHDF5
from shapely.geometry import Polygon
# endregion

#region Path definition
WorkPath = os.getcwd()
SimDir = "DeepCrLib"  
SimName = "Rectangle_Proton_0.0316_50_0_1"
simpath = "/Users/chiche/Desktop/DeepCrAnalysis/Simulations/DeepCrLibV1/"\
      + SimName + "_0.hdf5"
BatchID = "SurfaceAntennas"
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
Save = False
# Plot the surface footprint
PlotSurfaceFootprint(footprint, XmaxPos, Shower.zenith, Shower.azimuth, Save, BatchID)
# Compare the surface footprint with the in-ice footprint
CompareFootprints(footprint, Shower.zenith, Shower.azimuth, all_xray, all_yray, Save, BatchID)


# Plot the sampled surface footprint
plt.scatter(footprint_samples[:, 0], footprint_samples[:, 1], color='blue', s=1, label='Sampled points')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title(f'Sampled surface footprint ($\\theta$={Shower.zenith}°, $\\varphi$={Shower.azimuth}°)')
#plt.savefig(f"{OutputPath}_SampledFootprint.pdf", bbox_inches="tight") if Save else None
plt.show()
#sys.exit()

# Histogram of the time delay distribution
plt.hist(np.array(all_dt_samples)*1e9, bins=10, color='sandybrown', alpha=0.7, edgecolor='black')
plt.xlabel('Time [ns]')
plt.ylabel('Count')
plt.show()

# Amplitude dilution scatter plot
plt.scatter(all_xray_samples, all_yray_samples, c=DilutionFactor, cmap='jet', s=1, label='Ray paths')
cbar = plt.colorbar()
cbar.set_label('1/dL [$m^{-1}$]')
plt.show()

# Amplitude dilution scatter plot
plt.scatter(all_xray_samples, all_yray_samples, c=SurfaceDeepRatio*100, cmap='jet', s=1, label='Ray paths')
cbar = plt.colorbar()
cbar.set_label('$[\%]$')
plt.show()






