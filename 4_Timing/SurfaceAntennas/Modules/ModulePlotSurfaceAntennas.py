import numpy as np
import matplotlib.pyplot as plt


def PlotSurfaceFootprint(footprint, Xmax, zenith, azimuth, Save, BatchID):
    plt.figure(figsize=(6, 6))
    plt.plot(footprint[:, 0], footprint[:, 1], label='Surface footprint', color='blue')
    #plt.scatter(Xmax[0], Xmax[1], color='red', label='Xmax projection')
    plt.axis('equal')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title(f'(zenith={zenith}°, azimuth={azimuth}°)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{BatchID}_GroundFootprint.pdf", bbox_inches = "tight") if Save else None
    plt.show()


def CompareFootprints(footprint, zenith, azimuth, all_xray, all_yray, Save, BatchID):

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
    plt.savefig(f"{BatchID}_GroundvsDeepFootprint.pdf", bbox_inches = "tight") if Save else None
    plt.show()