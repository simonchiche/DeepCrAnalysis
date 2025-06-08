import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.path import Path

def cartesian_to_spherical_angles(uv):
    """
    Converts a unit vector (x, y, z) to spherical angles theta and phi.
    - theta: angle from the z-axis, in radians [0, pi]
    - phi: angle from the x-axis in the x-y plane, in radians [0, 2*pi)

    Assumes the vector is already normalized (unit vector).
    """
    x, y, z = uv[0], uv[1], uv[2]
    # theta = angle from z-axis
    theta = math.acos(z)

    # phi = angle from x-axis in xy-plane
    phi = math.atan2(y, x)
    if phi < 0:
        phi += 2 * math.pi

    return theta, phi

def GetShowerDirection(theta, phi):
    """Transforme des coordonnées sphériques en cartésiennes (unit vector)."""
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        -np.cos(theta)
    ])

def generate_footprint(Xmax, zenith_deg, azimuth_deg, theta_C_deg=1.0, n_rays=360):
    # Convert angles to radians
    zenith = np.radians(zenith_deg)
    azimuth = np.radians(azimuth_deg)
    theta_C = np.radians(theta_C_deg)

    # Direction de la gerbe (vecteur unitaire)
    shower_axis = GetShowerDirection(zenith, azimuth)
    #shower_axis[2] = -shower_axis[2]

    # Base locale orthonormée autour de l'axe de la gerbe
    z_axis = shower_axis
    # Trouver un vecteur perpendiculaire
    if np.allclose(z_axis, [0, 0, 1]):
        tmp = np.array([1, 0, 0])
    else:
        tmp = np.array([0, 0, 1])
    x_axis = np.cross(z_axis, tmp)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    # Échantillonnage d'angles autour du cône
    alphas = np.linspace(0, 2 * np.pi, n_rays)
    points = []

    for alpha in alphas:
        # Direction du rayon sur le cône de Cherenkov
        dir_vector = (
            np.cos(theta_C) * z_axis +
            np.sin(theta_C) * (np.cos(alpha) * x_axis + np.sin(alpha) * y_axis)
        )
        #print(dir_vector)

        # Intersecter le rayon avec le plan z = 0 (sol)
        dz = dir_vector[2]
        if dz == 0:
            continue  # rayon parallèle au sol, ne l’intersecte jamais
        t = -Xmax[2] / dz
        if t <= 0:
            print("test")
            continue  # rayon vers le haut

        point_on_ground = Xmax + t * dir_vector
        points.append(point_on_ground[:2])  # x, y

    return np.array(points)


def deg2rad(theta):
    return theta*np.pi/180.0

def n(z, model):
   
    depth = abs(z)
    if(model ==1): # Greenland
        A = 1.775
        if(z>-14.9):
            #print(z)
            B = -0.5019
            C =0.03247 
           
            n = A + B*np.exp(-C*depth)
        else:
            #print("here")
            B = -0.448023
            C = 0.02469 
            
            n = A + B*np.exp(-C*depth)
        
        if(z>0): n =1
        
    
    if(model == 2):
        #print("here")
        A = 1.775
        B = -0.43
        C =0.0132
        
        n = A + B*np.exp(-C*depth)
    if(z>0): n =1
        
    return n


def PropagateRay(SourcePos, Xmax, depth, glevel, model):

    ### Function that calculates the ray-bending path of a radio wave in ice ###
    # SourcePos: Initial position from which the ray is propagated
    # depth: depth of the ice sheet in m
    # theta: launch zenith angle in degrees
    # azimuth: launch azimuth angle in degrees
    # Xmax: In-air Xmax position
    # model: 1 for Greenland, 2 for South Pole

    # step size along the shower axis in m
    dl =1 
    # arrays to store the x and z positions, and the refractive index
    xray, yray, zray, nray, dt, dL = [SourcePos[0]], [SourcePos[1]], [SourcePos[2]], [1], [0], [0]
    z0 = SourcePos[2]
    tempDepth = z0 - glevel
    RayPos = SourcePos
    uray = SourcePos - Xmax
    uray = uray/np.sqrt(uray[0]**2 + uray[1]**2 + uray[2]**2)

    #print(tempDepth, "tempDepth")
    while(tempDepth>-depth):
        
        n1 = n(tempDepth, model)
        print(n1)
        RayPos = RayPos + uray*dl
        xray.append(RayPos[0])
        yray.append(RayPos[1])
        zray.append(RayPos[2])
        z = zray[-1]
        #print("z", z)
        tempDepth = z - glevel

        n2 = n(tempDepth, model)
        nray.append(n2)
        DeltaT = dt[-1] + dl/(3e8/n2) # time step in seconds
        dt.append(DeltaT)
        dL.append(dL[-1] + dl)
        itheta1, phi = cartesian_to_spherical_angles(uray)
        itheta2 = np.arcsin(n1/n2*np.sin(itheta1))
        uray = GetShowerDirection(itheta2, phi)
    
    #plt.plot(xray, zray, label='Ray Path', color='red')
    #plt.xlabel('x [m]')
    #plt.ylabel('z [m]')
    #plt.title('Ray Propagation in Ice')
    #plt.legend()
    #plt.grid(True)
    #plt.show()

    return xray, yray, zray, nray, dt, dL


def PropagateRayAll(SourcePosAll, Xmax, depth, glevel, model):
    """Propagate multiple rays from different source positions."""
    ### Function that calculates the ray-bending path of multiple radio waves in ice ###
    # SourcePosAll: List of initial positions from which the rays are propagated
    # Xmax: In-air Xmax position
    # depth: depth of the ice sheet in m
    # glevel: Ground level in the ice sheet
    # model: 1 for Greenland, 2 for South Pole
    all_xray, all_yray, all_zray, all_nray, all_dt, all_dL = [], [], [], [], [], []
    
    for SourcePos in SourcePosAll:
        SourcePos = np.array([SourcePos[0], SourcePos[1], glevel])  # Set z to 0 for propagation
        #print(SourcePos)
        # Propagate the ray from the current source position
        xray, yray, zray, nray, dt, dL = PropagateRay(SourcePos, Xmax, depth, glevel, model)
        all_xray.append(xray[-1])
        all_yray.append(yray[-1])
        all_zray.append(zray[-1])
        all_nray.append(nray[-1])
        all_dt.append(dt[-1])
        all_dL.append(dL[-1])
    

    return all_xray, all_yray, all_zray, all_nray, all_dt, all_dL


def sample_points_in_polygon(contour_points, N):
    """
    Generate N random points uniformly distributed within a polygon.
    
    contour_points: array-like, shape (M, 2) — the polygon (closed or not)
    N: number of points to generate
    """
    path = Path(contour_points)
    xmin, ymin = np.min(contour_points, axis=0)
    xmax, ymax = np.max(contour_points, axis=0)

    points = []
    while len(points) < N:
        x = np.random.uniform(xmin, xmax, size=N)
        y = np.random.uniform(ymin, ymax, size=N)
        candidates = np.vstack((x, y)).T
        inside = path.contains_points(candidates)
        accepted = candidates[inside]
        points.extend(accepted.tolist())

    return np.array(points[:N])
