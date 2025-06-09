#region Modules 
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os
from Modules.ModuleSurfaceAntennas import cartesian_to_spherical_angles, GetShowerDirection, generate_footprint, PropagateRayAll, sample_points_in_polygon, getXmaxPosition
from Modules.ModulePlotSurfaceAntennas import PlotSurfaceFootprint, CompareFootprints
from  MainModules.PlotConfig import MatplotlibConfig
from MainModules.ShowerClass import CreateShowerfromHDF5
# endregion

#region Path definition
WorkPath = os.getcwd()
SimDir = "DeepCrLib"  
SimName = "Rectangle_Proton_0.316_28_0_1"
simpath = "/Users/chiche/Desktop/DeepCrAnalysis/Simulations/DeepCrLibV1/"\
      + SimName + "_0.hdf5"
BatchID = "SurfaceAntennas"
OutputPath = MatplotlibConfig(WorkPath, SimDir, BatchID)
# endregion
Save = False
Shower = CreateShowerfromHDF5(simpath)

# =============================================================================

print(Shower.xmaxdist)
print(Shower.xmaxpos)

# region: Paramters
# Dxmax = distance from Xmax to the ground in meters
Dxmax = 768
 # ground level in meters
glevel = 3216 
# zenith angle in degrees
zenith = 28  
# azimuth angle in degrees (0 is towards North)
azimuth = 0.0  
# Xmax position
XmaxPos = getXmaxPosition(zenith, azimuth, glevel, Dxmax)
sys.exit()
# Cerenkov angle in degrees
theta_C = 1.2 
# footprint aperture angle in degrees
theta_lim = 3*theta_C  
# endregion 





# Surface footprint contours
footprint = \
    generate_footprint(XmaxPos, zenith, azimuth,theta_lim)

all_xray, all_yray, all_zray, all_nray, all_dt, all_dL =\
    PropagateRayAll(footprint, XmaxPos, 100, 3216, 1)

footprint_samples = sample_points_in_polygon(footprint, 1000)

all_xray_samples, all_yray_samples, all_zray_samples, all_nray_samples, all_dt_samples, all_dL_samples =\
    PropagateRayAll(footprint_samples, XmaxPos, 100, 3216, 1)


#######  PLOTS #########
# Display the surface footprint
Save = False
PlotSurfaceFootprint(footprint, XmaxPos, zenith, azimuth, Save, BatchID)
CompareFootprints(footprint, zenith, azimuth, all_xray, all_yray, Save, BatchID)



plt.scatter(footprint_samples[:, 0], footprint_samples[:, 1], color='blue', s=1, label='Sampled points')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title(f'Sampled surface footprint ($\\theta$={zenith}°, $\\varphi$={azimuth}°)')
plt.savefig(f"{OutputPath}_SampledFootprint.pdf", bbox_inches="tight") if Save else None
plt.show()
#sys.exit()


plt.hist(np.array(all_dt_samples)*1e9, bins=10, color='lightblue', alpha=0.7, edgecolor='black')
plt.xlabel('Time [ns]')
plt.ylabel('Count')
plt.show()



