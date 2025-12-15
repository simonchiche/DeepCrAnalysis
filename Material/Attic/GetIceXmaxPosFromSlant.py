import numpy as np
import matplotlib.pyplot as plt

def GetShowerDirection(theta, phi):
    """
    Convert angles from degrees to radians and return the direction vector.
    """
    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi)
    
    x = np.sin(theta_rad) * np.cos(phi_rad)
    y = np.sin(theta_rad) * np.sin(phi_rad)
    z = -np.cos(theta_rad)
    
    return np.array([x, y, z])

def GetIceXmaxPosFromSlant(theta, phi, XmaxSlant):

    # Calcultate Xmax position in ice from the shower direction and the Xmax slant depth

    shower_direction = GetShowerDirection(theta, phi)
    XmaxPos = XmaxSlant * shower_direction

    return XmaxPos


theta = 20  # Example angle in degrees
phi = 0    # Example azimuth angle in degrees
XmaxSlant = 5  # Example slant depth in meters

uv = GetShowerDirection(theta, phi)
print("Shower direction vector:", uv)
XmaxPos = GetIceXmaxPosFromSlant(theta, phi, XmaxSlant)
print("Xmax position in ice:", XmaxPos)