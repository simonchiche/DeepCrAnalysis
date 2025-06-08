import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from Modules.ModuleSurfaceAntennas import cartesian_to_spherical_angles, GetShowerDirection, generate_footprint, PropagateRayAll, sample_points_in_polygon


# Input parameters
# D = distance from Xmax to the ground in meters
D = 1200
# zenith angle in degrees
zenith = 28  
# Xmax position
Xmax = np.array([-D*np.sin(zenith*np.pi/180.0), 0.0, D*np.cos(zenith*np.pi/180.0) + 3216])  
# azimuth angle in degrees (0 is towards North)
azimuth = 0.0  
# Cerenkov angle in degrees
theta_C = 1.2 
# footprint aperture angle in degrees
theta_lim = 3*theta_C  

# Surface footprint contours
footprint = \
    generate_footprint(Xmax, zenith, azimuth,theta_lim)

all_xray, all_yray, all_zray, all_nray, all_dt, all_dL =\
    PropagateRayAll(footprint, Xmax, 100, 3216, 1)

# Display the surface footprint
plt.figure(figsize=(6, 6))
plt.plot(footprint[:, 0], footprint[:, 1], label='Footprint at ground', color='blue')
#plt.scatter(0, 0, color='red', label='Projection verticale de Xmax')
plt.axis('equal')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title(f'(zenith={zenith}°, azimuth={azimuth}°)')
plt.legend()
plt.grid(True)
plt.show()

# Display the footprint
plt.figure(figsize=(6, 6))
plt.plot(footprint[:, 0], footprint[:, 1], label='Surface footprint', color='blue')
plt.plot(all_xray, all_yray, color='red', label='In-ice footprint')
#plt.scatter(0, 0, color='red', label='Projection verticale de Xmax')
plt.axis('equal')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title(f'(zenith={zenith}°, azimuth={azimuth}°)')
plt.legend()
plt.grid(True)
plt.show()


samples = sample_points_in_polygon(footprint, 1000)

all_xray, all_yray, all_zray, all_nray, all_dt, all_dL =\
    PropagateRayAll(samples, Xmax, 100, 3216, 1)

plt.scatter(samples[:, 0], samples[:, 1], color='green', s=1, label='Sampled points')
plt.show()
#sys.exit()


plt.hist(np.array(all_dt)*1e9, bins=10, color='lightblue', alpha=0.7, edgecolor='black')
plt.xlabel('Time [ns]')
plt.ylabel('Count')
plt.show()



